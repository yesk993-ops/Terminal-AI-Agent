use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use colored::*;
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use regex::Regex;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use textwrap::wrap;
use tokio::time::timeout;

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
    /// The message text content.
    pub content: String,
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

// ---------------------------------------------------------------------------
// Generic API caller
// ---------------------------------------------------------------------------

/// Builds the request, sends it, and parses the response.
async fn call_api(
    client: &Client,
    url: &str,
    extra_headers: impl Fn(RequestBuilder) -> RequestBuilder,
    api_key: &str,
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
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json");
    let req = extra_headers(req);

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
        .map(|c| c.message.content)
        .ok_or(ApiError::NoChoices)
}

/// Calls the OpenRouter API for a single model.
///
/// Sends a chat completion request to `https://openrouter.ai/api/v1/chat/completions`
/// with the given model, messages, and temperature.
pub async fn call_openrouter(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
) -> Result<String, ApiError> {
    call_api(
        client,
        "https://openrouter.ai/api/v1/chat/completions",
        |r| {
            r.header("HTTP-Referer", "https://github.com/terminal-ai-agent")
                .header("X-Title", "Terminal AI Agent")
        },
        api_key,
        model,
        messages,
        temperature,
    )
    .await
}

/// Calls the Groq API for a single model.
///
/// Sends a chat completion request to `https://api.groq.com/openai/v1/chat/completions`
/// with the given model, messages, and temperature.
pub async fn call_groq(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
) -> Result<String, ApiError> {
    call_api(
        client,
        "https://api.groq.com/openai/v1/chat/completions",
        |r| r,
        api_key,
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

fn history_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let dir = PathBuf::from(&home).join(".local/share/terminal_ai_agent");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("history.json")
}

/// Persists the current conversation history to disk at `~/.local/share/terminal_ai_agent/history.json`.
pub fn save_conversation() {
    let path = history_path();
    if let Ok(data) = serde_json::to_string(&*conv()) {
        let _ = std::fs::write(&path, data);
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

/// Renders a model response into a bordered string with ANSI color codes.
///
/// * Strips markdown formatting (`**bold**`, `### headings`, `` `inline code` ``, `* list`)
/// * Applies colors: bold cyan for key terms, bold yellow for headings,
///   amber for code blocks, green for inline commands
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
                .map(|s| format!("\x1b[0;33m{}\x1b[0m", s))
                .collect();
            lines.extend(wrapped);
            continue;
        }

        let is_heading = trimmed.starts_with("### ")
            || trimmed.starts_with("## ")
            || trimmed.starts_with("# ");
        let is_shell_cmd = sc_re.is_match(raw_line);

        let raw_stripped = h_re.replace_all(raw_line, "");
        let raw_stripped = list_star_re().replace_all(&raw_stripped, "");

        let wrapped: Vec<String> = wrap(&raw_stripped, inner_w)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        for sub in wrapped {
            let mut processed = sub;
            if is_heading {
                processed = format!("\x1b[1;33m{}\x1b[0m", processed);
            }
            if is_shell_cmd {
                processed = format!("\x1b[0;33m{}\x1b[0m", processed);
            }
            processed = ic_re
                .replace_all(&processed, |caps: &regex::Captures| {
                    format!("\x1b[0;32m{}\x1b[0m", &caps[1])
                })
                .to_string();
            processed = b_re
                .replace_all(&processed, |caps: &regex::Captures| {
                    format!("\x1b[1;36m{}\x1b[0m", &caps[1])
                })
                .to_string();
            lines.push(processed);
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    let hline = "─".repeat(inner_w + 2);
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
        content: query.to_string(),
    };
    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: "You are an expert assistant. Provide clear, concise, and correct answers."
            .to_string(),
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

    let mut last_err = (String::new(), ApiError::Timeout);
    while let Some(result) = unordered.next().await {
        match result {
            Ok(resp) => {
                let assistant = ChatMessage {
                    role: "assistant".to_string(),
                    content: resp.clone(),
                };
                push_conversation(assistant);
                save_conversation();
                println!("{}", format_response(&resp));
                return;
            }
            Err(e) => last_err = e,
        }
    }

    // All failed
    conv().pop();
    eprintln!(
        "{} {}: {}",
        "All free models failed.".red(),
        last_err.0.cyan(),
        last_err.1.to_string().red()
    );
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
        assert!(result.starts_with("─"));
        assert!(result.ends_with("─"));
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
        assert!(result.contains("\x1b[1;36m"));
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
        assert!(result.starts_with("─"));
        assert!(result.ends_with("─"));
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
        let mut c = conv();
        c.clear();
        drop(c);

        let msg = ChatMessage {
            role: "user".to_string(),
            content: "first".to_string(),
        };
        push_conversation(msg);
        assert_eq!(conversation_history().len(), 1);

        // Push beyond max
        for i in 0..20 {
            push_conversation(ChatMessage {
                role: "user".to_string(),
                content: format!("msg {}", i),
            });
        }

        let hist = conversation_history();
        assert_eq!(hist.len(), 12);
        // Oldest messages should have been removed
        assert_eq!(hist[0].content, "msg 8");
        assert_eq!(hist[11].content, "msg 19");
    }
}
