use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use colored::*;
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use regex::Regex;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use textwrap::wrap;
use tokio::sync::RwLock;
use tokio::time::timeout;

pub mod tools;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when calling an AI model API.
#[derive(Debug)]
pub enum ApiError {
    /// Network-level failure (connection refused, DNS, etc.).
    Network(String),
    /// The API returned an unsuccessful HTTP status code.
    Http { status: u16, detail: String },
    /// The request timed out.
    Timeout,
    /// Failed to parse the API response body.
    Parse(String),
    /// The response contained no completion choices.
    NoChoices,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Network(e) => write!(f, "Network error: {}", e),
            ApiError::Http { status, detail } => {
                if detail.is_empty() {
                    write!(f, "HTTP {}", status)
                } else {
                    write!(f, "HTTP {} ({})", status, detail)
                }
            }
            ApiError::Timeout => write!(f, "Timeout"),
            ApiError::Parse(e) => write!(f, "Parse error: {}", e),
            ApiError::NoChoices => write!(f, "No choices returned"),
        }
    }
}

impl std::error::Error for ApiError {}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    /// Role of the message author: `"user"`, `"assistant"`, or `"system"`.
    pub role: String,
    /// The message text content (null for tool call messages).
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls made by the assistant (only present in assistant messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message is responding to (only present in tool messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool call as returned by the model.
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub type_field: String,
    /// The function details.
    pub function: FunctionCall,
}

/// A function call within a tool call.
#[derive(Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    /// Name of the function to call.
    pub name: String,
    /// JSON string of arguments.
    pub arguments: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

// ---------------------------------------------------------------------------
// Model lists & configuration
// ---------------------------------------------------------------------------

static FREE_MODELS: &[&str] = &[
    "deepseek/deepseek-chat-v3-0324:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen3-235b-a22b:free",
];

static GROQ_MODELS: &[&str] = &[
    "llama-3.3-70b-versatile",
];

static GOOGLE_MODELS: &[&str] = &[
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
];

static NVIDIA_MODELS: &[&str] = &[
    "nemotron-3-super",
    "meta/llama-3.1-8b-instruct",
];

/// Free models available through the local `opencode-to-openai` gateway
/// (no API key needed when the gateway is running).
static OPENCODE_GATEWAY_MODELS: &[&str] = &[
    "opencode/big-pickle",
    "opencode/gpt-5-nano",
    "opencode/minimax-m2.5-free",
    "opencode/nemotron-3-super-free",
];

/// Returns the OpenRouter API key from the `OPENROUTER_API_KEY` env var, or an empty string if unset.
pub fn get_openrouter_key() -> String {
    std::env::var("OPENROUTER_API_KEY").unwrap_or_default()
}

/// Returns the list of models to try on OpenRouter.
pub fn get_models() -> Vec<String> {
    if let Ok(m) = std::env::var("OPENROUTER_MODEL") {
        vec![m]
    } else {
        FREE_MODELS.iter().map(|s| s.to_string()).collect()
    }
}

/// Returns a Groq-compatible API key: tries GROQ_API_KEY, then OPENROUTER_API_KEY.
pub fn get_groq_key() -> Result<String, String> {
    std::env::var("GROQ_API_KEY")
        .map_err(|_| ())
        .or_else(|_| std::env::var("OPENROUTER_API_KEY").map_err(|_| ()))
        .map_err(|_| "GROQ_API_KEY or OPENROUTER_API_KEY not set".to_string())
}

/// Returns the Google Gemini API key from the `GOOGLE_API_KEY` env var, or empty if unset.
pub fn get_google_key() -> String {
    std::env::var("GOOGLE_API_KEY").unwrap_or_default()
}

/// Returns the NVIDIA NIM API key from the `NVIDIA_API_KEY` env var, or empty if unset.
pub fn get_nvidia_key() -> String {
    std::env::var("NVIDIA_API_KEY").unwrap_or_default()
}

/// Returns the base URL of the local `opencode-to-openai` gateway.
/// Defaults to `http://127.0.0.1:8083`. Override via `OPENCODE_GATEWAY_URL`.
pub fn get_opencode_gateway_url() -> String {
    std::env::var("OPENCODE_GATEWAY_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8083".to_string())
}

// ---------------------------------------------------------------------------
// Generic API caller
// ---------------------------------------------------------------------------

/// Builds the request, sends it, and parses the response.
///
/// The `headers` closure receives a bare `RequestBuilder` (with Content-Type and User-Agent
/// already set) and must add the Authorization header (or any provider-specific headers).
async fn call_api(
    client: &Client,
    url: &str,
    headers: impl Fn(RequestBuilder) -> RequestBuilder,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Result<String, ApiError> {
    let body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        temperature,
        max_tokens,
        stream: false,
    };

    let req = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "TerminalAI-Agent/0.1.0");
    let req = headers(req);

    let resp = req
        .json(&body)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let detail = http_status_detail(status);
        return Err(ApiError::Http {
            status,
            detail: detail.to_string(),
        });
    }

    let chat: ChatResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::Parse(e.to_string()))?;

    chat.choices
        .into_iter()
        .next()
        .map(|c| c.message.content.unwrap_or_default())
        .ok_or(ApiError::NoChoices)
}

/// Calls the OpenRouter API for a single model.
pub async fn call_openrouter(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://openrouter.ai/api/v1/chat/completions",
        move |r| {
            r.header("Authorization", format!("Bearer {}", k))
                .header("HTTP-Referer", "https://github.com/terminal-ai-agent")
                .header("X-Title", "Terminal AI Agent")
        },
        model,
        messages,
        temperature,
        max_tokens,
    )
    .await
}

/// Calls the Groq API for a single model.
pub async fn call_groq(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://api.groq.com/openai/v1/chat/completions",
        move |r| r.header("Authorization", format!("Bearer {}", k)),
        model,
        messages,
        temperature,
        max_tokens,
    )
    .await
}

/// Calls the Google Gemini API (OpenAI-compatible endpoint) for a single model.
///
/// Uses `x-goog-api-key` header instead of Bearer auth – Google API keys
/// are not accepted as Bearer tokens.
pub async fn call_google(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        move |r| r.header("x-goog-api-key", &k),
        model,
        messages,
        temperature,
        max_tokens,
    )
    .await
}

/// Calls the NVIDIA NIM API for a single model.
pub async fn call_nvidia(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://integrate.api.nvidia.com/v1/chat/completions",
        move |r| r.header("Authorization", format!("Bearer {}", k)),
        model,
        messages,
        temperature,
        max_tokens,
    )
    .await
}

/// Calls the local `opencode-to-openai` gateway (OpenAI-compatible).
///
/// No API key is needed by default; if the gateway requires one, set
/// `OPENCODE_GATEWAY_KEY`.
pub async fn call_opencode_gateway(
    client: &Client,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Result<String, ApiError> {
    let base = get_opencode_gateway_url();
    let url = format!("{}/v1/chat/completions", base);
    let gateway_key = std::env::var("OPENCODE_GATEWAY_KEY").unwrap_or_default();
    call_api(
        client,
        &url,
        |r| {
            if gateway_key.is_empty() {
                r
            } else {
                r.header("Authorization", format!("Bearer {}", &gateway_key))
            }
        },
        model,
        messages,
        temperature,
        max_tokens,
    )
    .await
}

// ---------------------------------------------------------------------------
// Conversation memory (RwLock + VecDeque for O(1) operations)
// ---------------------------------------------------------------------------

static CONVERSATION: LazyLock<RwLock<VecDeque<ChatMessage>>> =
    LazyLock::new(|| RwLock::new(VecDeque::new()));

const MAX_TURNS: usize = 12;

/// Appends a message to the in-memory conversation history.
///
/// Automatically trims history to the last 6 turns (12 messages) using VecDeque::pop_front (O(1)).
pub async fn push_conversation(msg: ChatMessage) {
    let mut c = CONVERSATION.write().await;
    c.push_back(msg);
    while c.len() > MAX_TURNS {
        c.pop_front();
    }
}

/// Returns a copy of the current conversation history.
pub async fn conversation_history() -> Vec<ChatMessage> {
    CONVERSATION.read().await.iter().cloned().collect()
}

/// Clears the conversation history.
pub async fn clear_conversation() {
    CONVERSATION.write().await.clear();
}

/// Cached history path — computed once, reused on every save/load.
fn history_path() -> &'static PathBuf {
    static PATH: LazyLock<PathBuf> = LazyLock::new(|| {
        let data_home = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(home).join(".local/share")
            });
        let dir = data_home.join("terminal_ai_agent");
        let _ = std::fs::create_dir_all(&dir);
        dir.join("history.json")
    });
    &PATH
}

/// Persists the current conversation history to disk (non-blocking).
pub async fn save_conversation() {
    let path = history_path().clone();
    let data = {
        let c = CONVERSATION.read().await;
        serde_json::to_string(&*c).ok()
    };
    if let Some(data) = data {
        tokio::task::spawn_blocking(move || {
            let _ = std::fs::write(&path, &data);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            }
        })
        .await
        .ok();
    }
}

/// Loads a previously saved conversation from disk (non-blocking).
pub async fn load_conversation() {
    let path = history_path().clone();
    let hist = tokio::task::spawn_blocking(move || {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|data| serde_json::from_str::<Vec<ChatMessage>>(&data).ok())
    })
    .await
    .ok()
    .flatten();
    if let Some(hist) = hist {
        let mut c = CONVERSATION.write().await;
        *c = VecDeque::from(hist);
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn safe_termwidth() -> usize {
    textwrap::termwidth().max(40)
}

fn http_status_detail(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized – check your API key",
        403 => "Forbidden",
        429 => "Rate limited – try again shortly",
        500 => "Provider error",
        502..=504 => "Provider unavailable",
        _ => "",
    }
}

fn bold_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*(.*?)\*\*").unwrap());
    &RE
}

fn heading_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^#{1,3}\s+").unwrap());
    &RE
}

fn list_star_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*\*\s+").unwrap());
    &RE
}

fn ansi_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*m").unwrap());
    &RE
}

fn inline_code_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`([^`]+)`").unwrap());
    &RE
}

fn shell_cmd_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\$\s+").unwrap());
    &RE
}

fn acronym_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[A-Z]{2,}\s*\([^)]+\)").unwrap());
    &RE
}

/// Renders a model response into a bordered string with ANSI color codes.
///
/// * Strips markdown formatting (`**bold**`, `### headings`, `` `inline code` ``, `* list`)
/// * Applies colors: bold green for key terms, green for headings,
///   gold for acronym definitions (e.g. `SLA (Service Level Agreement)`),
///   green for code blocks and inline commands
/// * Wraps text to terminal width
/// * Adds a horizontal rule top and bottom (no side walls) for easy copy-paste
pub fn format_response(resp: &str) -> String {
    let term_w = safe_termwidth();
    let inner_w = term_w.saturating_sub(2).max(20);

    let mut lines: Vec<String> = Vec::new();
    let b_re = bold_re();
    let h_re = heading_re();
    let ic_re = inline_code_re();
    let sc_re = shell_cmd_re();
    let ac_re = acronym_re();
    let a_re = ansi_re();
    let mut in_code = false;

    for raw_line in resp.lines() {
        let trimmed = raw_line.trim();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            continue;
        }

        if in_code {
            let wrapped: Vec<String> = wrap(raw_line, inner_w)
                .into_iter()
                .map(|s| format!("\x1b[0;32m{}\x1b[0m", s))
                .collect();
            lines.extend(wrapped);
            continue;
        }

        let is_heading = trimmed.starts_with("### ")
            || trimmed.starts_with("## ")
            || trimmed.starts_with("# ");
        let is_shell_cmd = sc_re.is_match(raw_line);
        let is_acronym = ac_re.is_match(raw_line);

        let raw_stripped = h_re.replace_all(raw_line, "");
        let raw_stripped = list_star_re().replace_all(&raw_stripped, "");

        let wrapped: Vec<String> = wrap(&raw_stripped, inner_w)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        for sub in wrapped {
            let mut processed = sub;
            if is_heading {
                processed = format!("\x1b[32m{}\x1b[0m", processed);
            }
            if is_shell_cmd {
                processed = format!("\x1b[0;32m{}\x1b[0m", processed);
            }
            processed = ic_re
                .replace_all(&processed, |caps: &regex::Captures| {
                    format!("\x1b[0;32m{}\x1b[0m", &caps[1])
                })
                .to_string();
            processed = b_re
                .replace_all(&processed, |caps: &regex::Captures| {
                    format!("\x1b[0;32m{}\x1b[0m", &caps[1])
                })
                .to_string();
            if is_acronym {
                // Strip all ANSI codes then re-wrap in gold so it's pure
                let stripped = a_re.replace_all(&processed, "").to_string();
                processed = format!("\x1b[38;5;220m{}\x1b[0m", stripped);
            }
            lines.push(processed);
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    let hline = format!("\x1b[36m{}\x1b[0m", "─".repeat(inner_w + 2));
    let mut out = String::new();
    out.push_str(&format!("{}\n", hline));
    for line in &lines {
        let stripped = a_re.replace_all(line, "").to_string();
        let pad = inner_w - stripped.chars().count().min(inner_w);
        out.push_str(&format!("{} {:pad$}\n", line, "", pad = pad));
    }
    out.push_str(&hline.to_string());
    out
}

// ---------------------------------------------------------------------------
// Quality helpers: temperature tuning, response scoring, post-processing
// ---------------------------------------------------------------------------

/// Detects the task type from the query and returns an optimized temperature.
///
/// Factual/analytical questions get lower temperature for accuracy.
/// Creative/code tasks get moderate temperature.
fn auto_temperature(query: &str) -> f32 {
    let lower = query.to_lowercase();
    let factual_keywords = [
        "what is", "who is", "when did", "where is", "how many",
        "define", "explain", "difference between", "compare",
        "version", "release date", "population", "capital",
    ];
    let creative_keywords = [
        "write", "create", "generate", "design", "imagine",
        "poem", "story", "haiku", "song", "brainstorm",
    ];
    let code_keywords = [
        "code", "function", "implement", "refactor", "debug",
        "error", "bug", "fix", "algorithm", "compile", "rust",
        "python", "javascript", "dockerfile", "git",
    ];

    if factual_keywords.iter().any(|k| lower.contains(k)) {
        0.2
    } else if code_keywords.iter().any(|k| lower.contains(k)) {
        0.4
    } else if creative_keywords.iter().any(|k| lower.contains(k)) {
        0.8
    } else {
        0.5
    }
}

/// Returns a default max_tokens value based on query complexity.
fn auto_max_tokens(query: &str) -> u32 {
    let word_count = query.split_whitespace().count();
    if word_count > 50 {
        4096
    } else if query.len() > 200 {
        2048
    } else {
        1024
    }
}

/// Scores a response for quality. Higher is better.
///
/// Factors: length (not too short, not too long), presence of structure
/// (headings, lists, code), and absence of refusals.
fn score_response(resp: &str) -> f64 {
    let len = resp.len() as f64;
    let lines = resp.lines().count() as f64;

    // Length score: penalize too short or too long
    let length_score = if len < 20.0 {
        len / 20.0 * 20.0
    } else if len < 100.0 {
        20.0 + (len - 20.0) / 80.0 * 30.0
    } else if len < 3000.0 {
        50.0
    } else {
        50.0 - ((len - 3000.0) / 3000.0 * 20.0).min(20.0)
    };

    // Structure score: headings, lists, code blocks, bold text
    let mut structure = 0.0;
    if resp.contains("### ") || resp.contains("## ") {
        structure += 15.0;
    }
    if resp.contains("- ") || resp.contains("* ") {
        structure += 10.0;
    }
    if resp.contains("```") {
        structure += 10.0;
    }
    if resp.contains("**") {
        structure += 5.0;
    }
    if lines > 3.0 {
        structure += 5.0;
    }

    // Penalty for refusals / low-effort
    let lower = resp.to_lowercase();
    let refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "as an ai", "i don't have", "i do not have",
        "sorry, i", "unfortunately, i",
    ];
    let refusal_penalty: f64 = refusal_phrases
        .iter()
        .filter(|p| lower.contains(*p))
        .map(|_| 30.0)
        .sum();

    (length_score + structure - refusal_penalty).max(0.0)
}

/// Post-processes a model response for display.
///
/// - Strips excessive blank lines
/// - Removes trailing whitespace
/// - Normalizes markdown list markers
/// - Removes model self-identification boilerplate
fn post_process_response(resp: &str) -> String {
    let mut result = String::with_capacity(resp.len());

    // Remove common boilerplate prefixes
    let stripped = resp
        .lines()
        .filter(|line| {
            let lower = line.trim().to_lowercase();
            !lower.starts_with("here is ") && !lower.starts_with("here are ")
                && !lower.starts_with("sure,") && !lower.starts_with("of course,")
                && !lower.starts_with("certainly,")
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Collapse 3+ blank lines into 2
    let mut blank_count = 0u32;
    for line in stripped.lines() {
        if line.trim().is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                result.push('\n');
            }
        } else {
            blank_count = 0;
            result.push_str(line.trim_end());
            result.push('\n');
        }
    }

    result.trim().to_string()
}

/// Summarizes old conversation messages to preserve key context.
///
/// Takes the oldest messages and compresses them into a summary message,
/// keeping the most recent messages intact for continuity.
fn summarize_old_context(messages: &[ChatMessage], keep_recent: usize) -> Vec<ChatMessage> {
    if messages.len() <= keep_recent {
        return messages.to_vec();
    }

    let old = &messages[..messages.len() - keep_recent];
    let recent = &messages[messages.len() - keep_recent..];

    // Extract key facts from old messages
    let mut summary_parts: Vec<String> = Vec::new();
    for msg in old {
        if let Some(ref content) = msg.content {
            let trimmed = content.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Keep first 100 chars of each old message as summary
            if trimmed.len() > 100 {
                summary_parts.push(format!("{}...", &trimmed[..100]));
            } else {
                summary_parts.push(trimmed.to_string());
            }
        }
    }

    if summary_parts.is_empty() {
        return recent.to_vec();
    }

    let summary_text = format!(
        "[Context from earlier conversation: {}]",
        summary_parts.join("; ")
    );

    let mut result = vec![ChatMessage {
        role: "system".to_string(),
        content: Some(summary_text),
        tool_calls: None,
        tool_call_id: None,
    }];
    result.extend_from_slice(recent);
    result
}

// ---------------------------------------------------------------------------
// Query processing
// ---------------------------------------------------------------------------

type ModelResult = Result<String, (String, ApiError)>;

/// Enhanced system prompt for chat mode — structured, detailed, high quality.
fn chat_system_prompt() -> String {
    r"You are a highly capable AI assistant. Follow these rules strictly:

## Response quality
- Lead with the direct answer, then explain if needed
- Use concrete examples, numbers, and specific facts — avoid vague generalities
- When comparing things, use a table or bullet points
- For technical topics, include code examples where relevant
- Keep responses focused — answer what was asked, nothing extra

## Formatting
- Use ### headings for major sections
- Use **bold** for key terms on first mention
- Use backtick-inline-code for commands, file paths, flags, and technical identifiers
- Use triple-backtick code blocks for multi-line code, configs, or shell commands
- Use - bullet points for lists of 3+ items
- Use numbered lists for sequential steps

## Personality
- Be direct and confident — don't hedge with hedging phrases
- Vary your examples and analogies each time
- Match the depth to the question: simple question = simple answer
- If the question is ambiguous, pick the most useful interpretation and answer it

## What NOT to do
- Don't start with filler phrases like Here is or Sure — just answer
- Don't repeat the question back
- Don't apologize or give disclaimers unless truly necessary
- Don't include unnecessary preamble or closing remarks"
        .to_string()
}

/// Builds a FuturesUnordered containing requests to all configured providers.
///
/// Each provider's models are raced in parallel; the caller iterates the stream
/// to pick the best response or handle tool-call loops.
fn build_provider_futures(
    client: &Client,
    messages: &Arc<Vec<ChatMessage>>,
    temperature: f32,
    max_tokens: u32,
    provider_timeout: Duration,
    opencode_timeout: Duration,
) -> FuturesUnordered<Pin<Box<dyn std::future::Future<Output = ModelResult> + Send>>> {
    let unordered: FuturesUnordered<
        Pin<Box<dyn std::future::Future<Output = ModelResult> + Send>>,
    > = FuturesUnordered::new();

    let ak = get_openrouter_key();
    let gk = get_groq_key();
    let gk_val = gk.clone().unwrap_or_default();
    let google_key = get_google_key();
    let nv_key = get_nvidia_key();
    let gw_key = std::env::var("OPENCODE_GATEWAY_KEY").unwrap_or_default();

    if !ak.is_empty() {
        let models = get_models();
        for m in models {
            let model = m;
            let cl = client.clone();
            let k = ak.clone();
            let msgs = Arc::clone(messages);
            unordered.push(Box::pin(async move {
                let res = timeout(
                    provider_timeout,
                    call_openrouter(&cl, &k, &model, &msgs, temperature, max_tokens),
                )
                .await;
                match res {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err((model, e)),
                    Err(_) => Err((model, ApiError::Timeout)),
                }
            }));
        }
    }

    if !gk_val.is_empty() {
        for m in GROQ_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let k = gk_val.clone();
            let msgs = Arc::clone(messages);
            unordered.push(Box::pin(async move {
                let res = timeout(
                    provider_timeout,
                    call_groq(&cl, &k, &model, &msgs, temperature, max_tokens),
                )
                .await;
                match res {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err((model, e)),
                    Err(_) => Err((model, ApiError::Timeout)),
                }
            }));
        }
    }

    if !google_key.is_empty() {
        for m in GOOGLE_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let k = google_key.clone();
            let msgs = Arc::clone(messages);
            unordered.push(Box::pin(async move {
                let res = timeout(
                    provider_timeout,
                    call_google(&cl, &k, &model, &msgs, temperature, max_tokens),
                )
                .await;
                match res {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err((model, e)),
                    Err(_) => Err((model, ApiError::Timeout)),
                }
            }));
        }
    }

    if !nv_key.is_empty() {
        for m in NVIDIA_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let k = nv_key.clone();
            let msgs = Arc::clone(messages);
            unordered.push(Box::pin(async move {
                let res = timeout(
                    provider_timeout,
                    call_nvidia(&cl, &k, &model, &msgs, temperature, max_tokens),
                )
                .await;
                match res {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err((model, e)),
                    Err(_) => Err((model, ApiError::Timeout)),
                }
            }));
        }
    }

    // OpenCode gateway
    for m in OPENCODE_GATEWAY_MODELS {
        let model = m.to_string();
        let cl = client.clone();
        let msgs = Arc::clone(messages);
        let gk_clone = gw_key.clone();
        unordered.push(Box::pin(async move {
            let base = get_opencode_gateway_url();
            let url = format!("{}/v1/chat/completions", base);
            let res = timeout(
                opencode_timeout,
                call_api(
                    &cl,
                    &url,
                    |r| {
                        if gk_clone.is_empty() {
                            r
                        } else {
                            r.header("Authorization", format!("Bearer {}", &gk_clone))
                        }
                    },
                    &model,
                    &msgs,
                    temperature,
                    max_tokens,
                ),
            )
            .await;
            match res {
                Ok(Ok(resp)) => Ok(resp),
                Ok(Err(e)) => Err((model, e)),
                Err(_) => Err((model, ApiError::Timeout)),
            }
        }));
    }

    unordered
}

/// Runs a user query against all available models in parallel with best-of-N selection.
///
/// Uses Arc for messages (zero-copy sharing), early cancellation when first good response arrives,
/// and auto-tuned temperature/max_tokens based on query analysis.
pub async fn process_query(
    client: &Client,
    query: &str,
    temperature: f32,
) {
    let effective_temp = if (temperature - 0.8).abs() < 0.01 {
        auto_temperature(query)
    } else {
        temperature
    };
    let max_tokens = auto_max_tokens(query);

    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: Some(query.to_string()),
        tool_calls: None,
        tool_call_id: None,
    };
    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: Some(chat_system_prompt()),
        tool_calls: None,
        tool_call_id: None,
    };

    push_conversation(user_msg.clone()).await;

    let raw_history = conversation_history().await;
    let history = summarize_old_context(&raw_history, 12);

    let mut msg_vec = vec![system_msg];
    msg_vec.extend(history);
    let messages = Arc::new(msg_vec);

    // Collect API keys once
    let ak = get_openrouter_key();
    let gk = get_groq_key();
    let google_key = get_google_key();
    let nv_key = get_nvidia_key();

    let mut unordered = build_provider_futures(
        client,
        &messages,
        effective_temp,
        max_tokens,
        Duration::from_secs(12),
        Duration::from_secs(8),
    );

    // Early cancellation: return as soon as we get a good response (score > 15.0)
    // or collect all and pick best
    let mut best_response: Option<String> = None;
    let mut best_score: f64 = 0.0;
    let mut attempts: Vec<String> = Vec::new();
    let mut all_responses: Vec<(String, f64)> = Vec::new();

    while let Some(result) = unordered.next().await {
        match result {
            Ok(resp) => {
                let processed = post_process_response(&resp);
                let score = score_response(&processed);
                all_responses.push((processed.clone(), score));

                if score > best_score {
                    best_score = score;
                    best_response = Some(processed);
                }
                // Early exit if we got a good enough response
                if best_score > 15.0 && best_response.is_some() {
                    break;
                }
            }
            Err(e) => {
                attempts.push(format!("{}: {}", e.0, e.1));
            }
        }
    }

    // If we broke early, don't drain - just drop the remaining futures
    // to avoid blocking on slow/timeout models

    if let Some(best) = best_response {
        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: Some(best.clone()),
            tool_calls: None,
            tool_call_id: None,
        };
        push_conversation(assistant).await;
        save_conversation().await;
        println!("{}", format_response(&best));
        return;
    }

    // All failed
    CONVERSATION.write().await.pop_back();

    let no_or = ak.is_empty();
    let no_groq = gk.is_err();
    let no_google = google_key.is_empty();
    let no_nvidia = nv_key.is_empty();
    let no_keys = no_or && no_groq && no_google && no_nvidia;

    eprintln!("{}", "All free models failed.".red());
    for a in &attempts {
        eprintln!("  {} {}", "•".yellow(), a.cyan());
    }
    if no_keys {
        eprintln!(
            "{}",
            "No API keys set. Export OPENROUTER_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or NVIDIA_API_KEY."
                .yellow()
        );
    } else {
        if no_groq {
            eprintln!(
                "{} {}",
                "Also set GROQ_API_KEY for faster fallback.".yellow(),
                "(console.groq.com/keys)".cyan()
            );
        }
        if no_google {
            eprintln!(
                "{} {}",
                "Also set GOOGLE_API_KEY for Gemini models.".yellow(),
                "(aistudio.google.com/apikey)".cyan()
            );
        }
        if no_nvidia {
            eprintln!(
                "{} {}",
                "Also set NVIDIA_API_KEY for NIM models.".yellow(),
                "(build.nvidia.com)".cyan()
            );
        }
    }
}

/// Enhanced system prompt for coding agent mode — detailed, structured, with error recovery.
fn coding_system_prompt() -> String {
    r#"You are an expert coding agent. You can read, write, edit files, run shell commands, search code, and find files.

## Available tools
- bash: Run shell commands. Args: { "command": "..." }
- read_file: Read file contents. Args: { "path": "..." }
- write_file: Create or overwrite a file. Args: { "path": "...", "content": "..." }
- edit_file: Replace text in a file (exact match). Args: { "path": "...", "old_string": "...", "new_string": "..." }
- grep: Search file contents with regex. Args: { "pattern": "...", "path": "...", "include": "..." }
- glob: Find files by pattern. Args: { "pattern": "...", "path": "..." }
- list_dir: List directory contents. Args: { "path": "..." }

## How to use tools
When you need a tool, respond with EXACTLY this format (one tag per tool call):
<tool_call>{"name":"TOOL_NAME","arguments":{...}}</tool_call>

Examples:
<tool_call>{"name":"bash","arguments":{"command":"ls -la /tmp"}}</tool_call>
<tool_call>{"name":"read_file","arguments":{"path":"src/main.rs"}}</tool_call>
<tool_call>{"name":"grep","arguments":{"pattern":"fn main","path":"src","include":"*.rs"}}</tool_call>

## Workflow
1. Understand the task — read relevant files first if needed
2. Plan your approach before acting
3. Call one tool at a time, wait for the result
4. Verify your work — check files after editing, run tests after changes
5. If a tool call fails, analyze the error and try a different approach
6. When done, provide a clear summary of what was done

## Error recovery
- If a file doesn't exist, check the path with list_dir or glob
- If a command fails, read the error message carefully and fix the issue
- If edit_file fails (old_string not found), read the file first to find the exact text
- If you're stuck, try a different approach — don't repeat the same failing action

## Rules
- Always use absolute paths
- Read files before editing them
- After writing/editing a file, verify the change if possible
- Don't describe what you would do — just do it
- Keep tool call arguments as JSON strings"#
        .to_string()
}

/// Runs a query in coding-agent mode with text-based tool calling.
///
/// Uses Arc for messages (zero-copy sharing), early cancellation, and
/// a special prompt for tool calling via text tags.
pub async fn process_code_query(
    client: &Client,
    query: &str,
    temperature: f32,
) {
    let effective_temp = if (temperature - 0.8).abs() < 0.01 {
        0.3
    } else {
        temperature
    };
    let max_tokens = 2048u32;

    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: Some(coding_system_prompt()),
        tool_calls: None,
        tool_call_id: None,
    };
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: Some(query.to_string()),
        tool_calls: None,
        tool_call_id: None,
    };

    let mut messages = vec![system_msg, user_msg];
    let tc_re = tool_call_re();
    let mut iterations = 0;
    const MAX_ITER: usize = 25;

    loop {
        iterations += 1;
        if iterations > MAX_ITER {
            eprintln!("{}", "[code-agent] Max iterations reached.".red());
            break;
        }

        let messages_arc = Arc::new(messages.clone());

        let mut unordered = build_provider_futures(
            client,
            &messages_arc,
            effective_temp,
            max_tokens,
            Duration::from_secs(20),
            Duration::from_secs(15),
        );

        // Early cancellation: return on first good response (score > 40)
        let mut best_resp: Option<String> = None;
        let mut best_score: f64 = -1.0;
        let mut attempts: Vec<String> = Vec::new();

        while let Some(result) = unordered.next().await {
            match result {
                Ok(resp) => {
                    let score = score_response(&resp);
                    if score > best_score {
                        best_score = score;
                        best_resp = Some(resp);
                    }
                    // Early exit for good enough responses
                    if best_score > 40.0 {
                        break;
                    }
                }
                Err((model, err)) => {
                    attempts.push(format!("{}: {}", model, err));
                }
            }
        }

        // Drain remaining futures
        if best_resp.is_some() {
            while unordered.next().await.is_some() {}
        }

        let resp = match best_resp {
            Some(r) => r,
            None => {
                eprintln!("{}", "[code-agent] All models failed.".red());
                for a in &attempts {
                    eprintln!("  {} {}", "•".yellow(), a.cyan());
                }
                break;
            }
        };

        // Check for text-based tool call
        let tool_call = tc_re.captures(&resp).and_then(|caps| {
            let tag = caps.get(1)?.as_str();
            let after_tag = &resp[caps.get(0).unwrap().end()..];
            let json_str = extract_balanced_json(after_tag)?;
            if tag == "tool_call" {
                serde_json::from_str::<ToolCallText>(json_str).ok()
            } else {
                let args: Value = serde_json::from_str(json_str).ok()?;
                Some(ToolCallText {
                    name: tag.to_string(),
                    arguments: args,
                })
            }
        });

        if let Some(tc) = tool_call {
            eprintln!(
                "{} {} {}",
                "[code-agent]".cyan(),
                "Running:".dimmed(),
                format!("{} {}", tc.name, summarize_args(&tc.arguments)).yellow()
            );

            messages = Arc::try_unwrap(messages_arc).unwrap_or_else(|arc| (*arc).clone());
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(resp),
                tool_calls: None,
                tool_call_id: None,
            });

            let result = tools::execute_tool(&tc.name, &tc.arguments).await;

            messages.push(ChatMessage {
                role: "user".to_string(),
                content: Some(format!("[tool result for {}]:\n{}", tc.name, result)),
                tool_calls: None,
                tool_call_id: None,
            });
        } else {
            println!("{}", format_response(&resp));
            return;
        }
    }
}

#[derive(Deserialize)]
struct ToolCallText {
    name: String,
    arguments: Value,
}

/// Creates a short human-readable summary of tool arguments for progress display.
fn summarize_args(args: &Value) -> String {
    if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
        let short: String = cmd.chars().take(60).collect();
        if cmd.len() > 60 {
            format!("`{}…`", short)
        } else {
            format!("`{}`", short)
        }
    } else if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
        path.to_string()
    } else if let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) {
        pattern.to_string()
    } else {
        String::new()
    }
}

fn tool_call_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"<(tool_call|bash|read_file|write_file|edit_file|grep|glob|list_dir)>\s*").unwrap()
    });
    &RE
}

/// Extracts a balanced JSON object from `s` starting at the first `{`.
/// Returns the slice covering the outermost `{...}` pair, handling nested
/// braces and strings correctly.
fn extract_balanced_json(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes[start..].iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && in_string {
            escaped = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(&s[start..start + i + 1]);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- format_response tests --

    #[test]
    fn test_format_response_plain_text() {
        let result = format_response("Hello world");
        assert!(result.contains("Hello world"));
        assert!(result.contains("─"));
    }

    #[test]
    fn test_format_response_strips_code_fences() {
        let result = format_response("```\ncode\n```");
        assert!(!result.contains("```"));
        assert!(result.contains("code"));
    }

    #[test]
    fn test_format_response_bold_markers_removed() {
        let result = format_response("this is **bold** text");
        assert!(!result.contains("**bold**"));
        assert!(result.contains("bold"));
        // ANSI escape inserted
        assert!(result.contains("\x1b[0;32m"));
        assert!(result.contains("\x1b[0m"));
    }

    #[test]
    fn test_format_response_heading_prefix_stripped() {
        let result = format_response("### Title");
        assert!(!result.contains("###"));
        assert!(result.contains("Title"));
    }

    #[test]
    fn test_format_response_inline_code_colored() {
        let result = format_response("run `cmd` now");
        assert!(!result.contains("`cmd`"));
        assert!(result.contains("cmd"));
        assert!(result.contains("\x1b[0;32m"));
    }

    #[test]
    fn test_format_response_empty() {
        let result = format_response("");
        assert!(result.contains("─"));
    }

    // -- http_status_detail tests --

    #[test]
    fn test_http_401_detail() {
        assert!(http_status_detail(401).contains("API key"));
    }

    #[test]
    fn test_http_429_detail() {
        assert!(http_status_detail(429).contains("Rate limited"));
    }

    #[test]
    fn test_http_unknown_detail() {
        assert_eq!(http_status_detail(418), "");
    }

    // -- safe_termwidth tests --

    #[test]
    fn test_safe_termwidth_minimum() {
        assert!(safe_termwidth() >= 40);
    }

    // -- get_openrouter_key (env) --

    #[test]
    fn test_get_openrouter_key_default_empty() {
        // Without env var it should return empty
        std::env::remove_var("OPENROUTER_API_KEY");
        assert_eq!(get_openrouter_key(), "");
    }

    // -- conversation tests (async) --

    #[tokio::test]
    async fn test_conversation_push_max_and_history() {
        clear_conversation().await;

        let msg = ChatMessage {
            role: "user".to_string(),
            content: Some("first".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        push_conversation(msg).await;
        assert_eq!(conversation_history().await.len(), 1);

        // Push beyond max
        for i in 0..20 {
            push_conversation(ChatMessage {
                role: "user".to_string(),
                content: Some(format!("msg {}", i)),
                tool_calls: None,
                tool_call_id: None,
            })
            .await;
        }

        let hist = conversation_history().await;
        assert_eq!(hist.len(), 12);
        // Oldest messages should have been removed
        assert_eq!(hist[0].content.as_deref(), Some("msg 8"));
        assert_eq!(hist[11].content.as_deref(), Some("msg 19"));
    }

    // -- auto_temperature tests --

    #[test]
    fn test_auto_temperature_factual() {
        assert!(auto_temperature("what is the capital of France") <= 0.3);
    }

    #[test]
    fn test_auto_temperature_creative() {
        assert!(auto_temperature("write a poem about the ocean") >= 0.7);
    }

    #[test]
    fn test_auto_temperature_code() {
        assert!(auto_temperature("implement a binary search in Rust") <= 0.5);
    }

    #[test]
    fn test_auto_temperature_default() {
        let temp = auto_temperature("tell me something interesting");
        assert!((0.3..=0.8).contains(&temp));
    }

    // -- auto_max_tokens tests --

    #[test]
    fn test_auto_max_tokens_short_query() {
        assert_eq!(auto_max_tokens("hi"), 1024);
    }

    #[test]
    fn test_auto_max_tokens_long_query() {
        assert!(auto_max_tokens(&"word ".repeat(60)) >= 2048);
    }

    // -- score_response tests --

    #[test]
    fn test_score_response_empty() {
        assert_eq!(score_response(""), 0.0);
    }

    #[test]
    fn test_score_response_short() {
        let score = score_response("Yes");
        assert!(score > 0.0 && score < 30.0);
    }

    #[test]
    fn test_score_response_structured() {
        let resp = "### Heading\n\n- item 1\n- item 2\n\n```\ncode\n```";
        let score = score_response(resp);
        assert!(score > 50.0);
    }

    #[test]
    fn test_score_response_refusal_penalty() {
        let resp = "I cannot help with that request.";
        let score = score_response(resp);
        assert!(score < 10.0);
    }

    // -- post_process_response tests --

    #[test]
    fn test_post_process_strips_filler() {
        let resp = "Here is the answer:\nFoo bar";
        let result = post_process_response(resp);
        assert!(!result.starts_with("Here is"));
        assert!(result.contains("Foo bar"));
    }

    #[test]
    fn test_post_process_collapses_blanks() {
        let resp = "line1\n\n\n\n\n\n\nline2";
        let result = post_process_response(resp);
        // Should collapse 6 blank lines down to 2
        assert!(!result.contains("\n\n\n\n"));
    }

    // -- summarize_old_context tests --

    #[test]
    fn test_summarize_old_context_no_op() {
        let msgs = vec![
            ChatMessage { role: "user".into(), content: Some("a".into()), tool_calls: None, tool_call_id: None },
            ChatMessage { role: "assistant".into(), content: Some("b".into()), tool_calls: None, tool_call_id: None },
        ];
        let result = summarize_old_context(&msgs, 5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_summarize_old_context_compression() {
        let msgs: Vec<ChatMessage> = (0..20)
            .map(|i| ChatMessage {
                role: if i % 2 == 0 { "user".into() } else { "assistant".into() },
                content: Some(format!("message {}", i)),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();
        let result = summarize_old_context(&msgs, 4);
        // Should have summary + 4 recent = 5
        assert!(result.len() <= 5);
        // Recent messages preserved
        assert_eq!(result.last().unwrap().content.as_deref(), Some("message 19"));
    }
}
