use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
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
    "nvidia/nemotron-3-super:free",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen3-235b-a22b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "microsoft/phi-4-reasoning:free",
    "openrouter/owl-alpha",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
];

static GROQ_MODELS: &[&str] = &[
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
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
) -> Result<String, ApiError> {
    let body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        temperature,
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
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://api.groq.com/openai/v1/chat/completions",
        move |r| r.header("Authorization", format!("Bearer {}", k)),
        model,
        messages,
        temperature,
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
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        move |r| r.header("x-goog-api-key", &k),
        model,
        messages,
        temperature,
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
) -> Result<String, ApiError> {
    let k = api_key.to_string();
    call_api(
        client,
        "https://integrate.api.nvidia.com/v1/chat/completions",
        move |r| r.header("Authorization", format!("Bearer {}", k)),
        model,
        messages,
        temperature,
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
    )
    .await
}

// ---------------------------------------------------------------------------
// Conversation memory
// ---------------------------------------------------------------------------

static CONVERSATION: LazyLock<Mutex<Vec<ChatMessage>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

fn conv() -> std::sync::MutexGuard<'static, Vec<ChatMessage>> {
    CONVERSATION.lock().unwrap_or_else(|e| e.into_inner())
}

/// Appends a message to the in-memory conversation history.
///
/// Automatically trims history to the last 6 turns (12 messages) to keep context size manageable.
pub fn push_conversation(msg: ChatMessage) {
    let mut c = conv();
    c.push(msg);
    const MAX_TURNS: usize = 12;
    while c.len() > MAX_TURNS {
        c.remove(0);
    }
}

/// Returns a copy of the current conversation history.
pub fn conversation_history() -> Vec<ChatMessage> {
    conv().clone()
}

/// Clears the conversation history.
pub fn clear_conversation() {
    conv().clear();
}

fn history_path() -> PathBuf {
    let data_home = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".local/share")
        });
    let dir = data_home.join("terminal_ai_agent");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("history.json")
}

/// Persists the current conversation history to disk with restrictive permissions (600).
pub fn save_conversation() {
    let path = history_path();
    if let Ok(data) = serde_json::to_string(&*conv()) {
        if let Ok(()) = std::fs::write(&path, &data) {
            #[cfg(unix)]
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
    }
}

/// Loads a previously saved conversation from disk, replacing the current in-memory history.
pub fn load_conversation() {
    let path = history_path();
    if let Ok(data) = std::fs::read_to_string(&path) {
        if let Ok(hist) = serde_json::from_str::<Vec<ChatMessage>>(&data) {
            let mut c = conv();
            *c = hist;
        }
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
                    format!("\x1b[1;32m{}\x1b[0m", &caps[1])
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
// Query processing
// ---------------------------------------------------------------------------

type ModelResult = Result<String, (String, ApiError)>;

/// Runs a user query against all available models (OpenRouter + Groq) in parallel.
///
/// The first model to return a valid response wins. The answer is printed to stdout
/// with colored formatting. On total failure, an error message is printed to stderr.
/// The user message is appended to conversation history before the call, and
/// the assistant response is appended on success.
pub async fn process_query(
    client: &Client,
    query: &str,
    temperature: f32,
) {
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: Some(query.to_string()),
        tool_calls: None,
        tool_call_id: None,
    };
    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: Some(
            "You are an expert assistant. Provide clear, concise, and correct answers. Vary your explanations each time — use different examples, analogies, or angles so the same question gets a fresh perspective."
                .to_string(),
        ),
        tool_calls: None,
        tool_call_id: None,
    };

    push_conversation(user_msg.clone());

    let mut messages = vec![system_msg];
    messages.extend(conversation_history().iter().cloned());

    let mut unordered: FuturesUnordered<
        Pin<Box<dyn std::future::Future<Output = ModelResult> + Send>>,
    > = FuturesUnordered::new();

    let ak = get_openrouter_key();
    if !ak.is_empty() {
        for m in get_models() {
            let model = m;
            let cl = client.clone();
            let k = ak.clone();
            let msgs = messages.clone();
            unordered.push(Box::pin(async move {
                let res = timeout(
                    Duration::from_secs(10),
                    call_openrouter(&cl, &k, &model, &msgs, temperature),
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

    if let Ok(gk) = get_groq_key() {
        for m in GROQ_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let k = gk.clone();
            let msgs = messages.clone();
            unordered.push(Box::pin(async move {
                let res = timeout(
                    Duration::from_secs(10),
                    call_groq(&cl, &k, &model, &msgs, temperature),
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

    let gk = get_google_key();
    if !gk.is_empty() {
        for m in GOOGLE_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let k = gk.clone();
            let msgs = messages.clone();
            unordered.push(Box::pin(async move {
                let res = timeout(
                    Duration::from_secs(10),
                    call_google(&cl, &k, &model, &msgs, temperature),
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

    let nk = get_nvidia_key();
    if !nk.is_empty() {
        for m in NVIDIA_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let k = nk.clone();
            let msgs = messages.clone();
            unordered.push(Box::pin(async move {
                let res = timeout(
                    Duration::from_secs(10),
                    call_nvidia(&cl, &k, &model, &msgs, temperature),
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

    // OpenCode gateway (no API key needed if running locally)
    for m in OPENCODE_GATEWAY_MODELS {
        let model = m.to_string();
        let cl = client.clone();
        let msgs = messages.clone();
        let gateway_key = std::env::var("OPENCODE_GATEWAY_KEY").unwrap_or_default();
        unordered.push(Box::pin(async move {
            let base = get_opencode_gateway_url();
            let url = format!("{}/v1/chat/completions", base);
            let res = timeout(
                Duration::from_secs(5),
                call_api(
                    &cl,
                    &url,
                    |r| {
                        if gateway_key.is_empty() {
                            r
                        } else {
                            r.header("Authorization", format!("Bearer {}", &gateway_key))
                        }
                    },
                    &model,
                    &msgs,
                    temperature,
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

    let mut attempts: Vec<String> = Vec::new();

    while let Some(result) = unordered.next().await {
        match result {
            Ok(resp) => {
                let assistant = ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(resp.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                };
                push_conversation(assistant);
                save_conversation();
                println!("{}", format_response(&resp));
                return;
            }
            Err(e) => {
                attempts.push(format!("{}: {}", e.0, e.1));
            }
        }
    }

    // All failed — show a helpful summary
    conv().pop();

    let no_or = get_openrouter_key().is_empty();
    let no_groq = get_groq_key().is_err();
    let no_google = get_google_key().is_empty();
    let no_nvidia = get_nvidia_key().is_empty();
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

/// Runs a query in coding-agent mode with text-based tool calling.
///
/// Uses a special prompt instructing the model to output tool calls in
/// `<tool_call>{"name":"...","arguments":{...}}</tool_call>` format.
/// This works with any text model (no API-level function calling required).
pub async fn process_code_query(
    client: &Client,
    query: &str,
    temperature: f32,
) {
    let sys_prompt = format!(
        "You are a coding agent with access to these tools:\n\
         - bash: Run a bash command. Args: {{ \"command\": \"...\" }}\n\
         - read_file: Read a file. Args: {{ \"path\": \"...\" }}\n\
         - write_file: Write a file. Args: {{ \"path\": \"...\", \"content\": \"...\" }}\n\
         - edit_file: Edit a file. Args: {{ \"path\": \"...\", \"old_string\": \"...\", \"new_string\": \"...\" }}\n\
         - grep: Search code. Args: {{ \"pattern\": \"...\", \"path\": \"...\", \"include\": \"...\" }}\n\
         - glob: Find files. Args: {{ \"pattern\": \"...\", \"path\": \"...\" }}\n\
         - list_dir: List directory. Args: {{ \"path\": \"...\" }}\n\n\
         When you need to use a tool, respond with EXACTLY:\n\
         <tool_call>{{ \"name\": \"TOOL_NAME\", \"arguments\": {{...}} }}</tool_call>\n\n\
         I will execute the tool and tell you the result. Then you can continue.\n\
         When the task is complete, respond with a brief summary (no tool calls).\n\
         Do NOT describe what you would do -- just call the tool."
    );

    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: Some(sys_prompt),
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

        let mut unordered: FuturesUnordered<
            Pin<Box<dyn std::future::Future<Output = ModelResult> + Send>>,
        > = FuturesUnordered::new();

        let ak = get_openrouter_key();
        if !ak.is_empty() {
            for m in get_models() {
                let model = m;
                let cl = client.clone();
                let k = ak.clone();
                let msgs = messages.clone();
                let m2 = model.clone();
                unordered.push(Box::pin(async move {
                    let res = timeout(
                        Duration::from_secs(15),
                        call_openrouter(&cl, &k, &model, &msgs, temperature),
                    )
                    .await;
                    match res {
                        Ok(Ok(resp)) => Ok(resp),
                        Ok(Err(e)) => Err((model, e)),
                        Err(_) => Err((m2, ApiError::Timeout)),
                    }
                }));
            }
        }

        if let Ok(gk) = get_groq_key() {
            for m in GROQ_MODELS {
                let model = m.to_string();
                let cl = client.clone();
                let k = gk.clone();
                let msgs = messages.clone();
                let m2 = model.clone();
                unordered.push(Box::pin(async move {
                    let res = timeout(
                        Duration::from_secs(15),
                        call_groq(&cl, &k, &model, &msgs, temperature),
                    )
                    .await;
                    match res {
                        Ok(Ok(resp)) => Ok(resp),
                        Ok(Err(e)) => Err((model, e)),
                        Err(_) => Err((m2, ApiError::Timeout)),
                    }
                }));
            }
        }

        let gk = get_google_key();
        if !gk.is_empty() {
            for m in GOOGLE_MODELS {
                let model = m.to_string();
                let cl = client.clone();
                let k = gk.clone();
                let msgs = messages.clone();
                let m2 = model.clone();
                unordered.push(Box::pin(async move {
                    let res = timeout(
                        Duration::from_secs(15),
                        call_google(&cl, &k, &model, &msgs, temperature),
                    )
                    .await;
                    match res {
                        Ok(Ok(resp)) => Ok(resp),
                        Ok(Err(e)) => Err((model, e)),
                        Err(_) => Err((m2, ApiError::Timeout)),
                    }
                }));
            }
        }

        let nk = get_nvidia_key();
        if !nk.is_empty() {
            for m in NVIDIA_MODELS {
                let model = m.to_string();
                let cl = client.clone();
                let k = nk.clone();
                let msgs = messages.clone();
                let m2 = model.clone();
                unordered.push(Box::pin(async move {
                    let res = timeout(
                        Duration::from_secs(15),
                        call_nvidia(&cl, &k, &model, &msgs, temperature),
                    )
                    .await;
                    match res {
                        Ok(Ok(resp)) => Ok(resp),
                        Ok(Err(e)) => Err((model, e)),
                        Err(_) => Err((m2, ApiError::Timeout)),
                    }
                }));
            }
        }

        // OpenCode gateway (no API key needed if running locally)
        for m in OPENCODE_GATEWAY_MODELS {
            let model = m.to_string();
            let cl = client.clone();
            let msgs = messages.clone();
            let m2 = model.clone();
            let gateway_key = std::env::var("OPENCODE_GATEWAY_KEY").unwrap_or_default();
            unordered.push(Box::pin(async move {
                let base = get_opencode_gateway_url();
                let url = format!("{}/v1/chat/completions", base);
                let res = timeout(
                    Duration::from_secs(10),
                    call_api(
                        &cl,
                        &url,
                        |r| {
                            if gateway_key.is_empty() {
                                r
                            } else {
                                r.header("Authorization", format!("Bearer {}", &gateway_key))
                            }
                        },
                        &model,
                        &msgs,
                        temperature,
                    ),
                )
                .await;
                match res {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err((m2, e)),
                    Err(_) => Err((m2, ApiError::Timeout)),
                }
            }));
        }

        let mut attempts: Vec<String> = Vec::new();
        let mut response: Option<String> = None;
        while let Some(result) = unordered.next().await {
            match result {
                Ok(resp) => {
                    response = Some(resp);
                    break;
                }
                Err((model, err)) => {
                    attempts.push(format!("{}: {}", model, err));
                }
            }
        }

        let resp = match response {
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
            let json_str = caps.get(2)?.as_str();
            if tag == "tool_call" {
                // <tool_call>{"name":"x","arguments":{...}}</tool_call>
                serde_json::from_str::<ToolCallText>(json_str).ok()
            } else {
                // <bash>{"command":"..."}</bash>  or  <write_file>{"path":"...","content":"..."}</write_file>
                let args: Value = serde_json::from_str(json_str).ok()?;
                Some(ToolCallText {
                    name: tag.to_string(),
                    arguments: args,
                })
            }
        });

        if let Some(tc) = tool_call {
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
            // No tool call — final answer
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

fn tool_call_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"<(tool_call|bash|read_file|write_file|edit_file|grep|glob|list_dir)>\s*(\{.*?\})\s*</[a-z_]+>").unwrap()
    });
    &RE
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
        assert!(result.contains("\x1b[1;32m"));
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

    // -- conversation tests (run serially via single test) --

    #[test]
    fn test_conversation_push_max_and_history() {
        clear_conversation();

        let msg = ChatMessage {
            role: "user".to_string(),
            content: Some("first".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        push_conversation(msg);
        assert_eq!(conversation_history().len(), 1);

        // Push beyond max
        for i in 0..20 {
            push_conversation(ChatMessage {
                role: "user".to_string(),
                content: Some(format!("msg {}", i)),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        let hist = conversation_history();
        assert_eq!(hist.len(), 12);
        // Oldest messages should have been removed
        assert_eq!(hist[0].content.as_deref(), Some("msg 8"));
        assert_eq!(hist[11].content.as_deref(), Some("msg 19"));
    }
}
