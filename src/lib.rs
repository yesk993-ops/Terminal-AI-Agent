use std::collections::{HashMap, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use colored::*;
use regex::Regex;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
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

#[derive(Serialize, Clone)]
pub struct Tool {
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone)]
pub enum ToolChoice {
    Auto,
    None,
    Function { r#type: String, function: serde_json::Value },
}

impl serde::Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value = match self {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Function { r#type, function } => {
                serde_json::json!({"type": r#type, "function": function})
            }
        };
        value.serialize(serializer)
    }
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
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
];

static GROQ_MODELS: &[&str] = &[
    "llama-3.3-70b-versatile",
];

/// Production models via NVIDIA NIM API — same as Python agent uses.
/// These have 1000+ RPM rate limits (no 429s in practice).
static NVIDIA_MODELS: &[&str] = &[
    "meta/llama-3.1-8b-instruct",
    "deepseek-ai/deepseek-v4-pro",
    "openai/gpt-oss-120b",
    "mistralai/mistral-small-4-119b-2603",
    "nvidia/nemotron-3-super-120b-a12b",
];

/// Qwen model via NVIDIA NIM — uses its own dedicated API key.
static NVIDIA_QWEN_MODELS: &[&str] = &[
    "qwen/qwen3.5-122b-a10b",
];

/// Returns the OpenRouter API key from the `OPENROUTER_API_KEY` env var, or an empty string if unset.
pub fn get_openrouter_key() -> String {
    std::env::var("OPENROUTER_API_KEY").unwrap_or_default()
}

/// Returns the list of models to try on OpenRouter.
pub fn get_models() -> &'static Vec<String> {
    static CACHED: LazyLock<Vec<String>> = LazyLock::new(|| {
        if let Ok(m) = std::env::var("OPENROUTER_MODEL") {
            vec![m]
        } else {
            FREE_MODELS.iter().map(|s| s.to_string()).collect()
        }
    });
    &CACHED
}

/// Returns a Groq-compatible API key. Only returns the key if GROQ_API_KEY
/// is explicitly set — does NOT fall back to OpenRouter key (different provider).
pub fn get_groq_key() -> Result<String, String> {
    std::env::var("GROQ_API_KEY")
        .map_err(|_| "GROQ_API_KEY not set".to_string())
}

/// Returns the NVIDIA NIM API key from the `NVIDIA_API_KEY` env var.
pub fn get_nvidia_key() -> String {
    std::env::var("NVIDIA_API_KEY").unwrap_or_default()
}

/// Returns the dedicated NVIDIA Qwen API key from the `NVIDIA_QWEN_API_KEY` env var.
pub fn get_nvidia_qwen_key() -> String {
    std::env::var("NVIDIA_QWEN_API_KEY").unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Generic API caller
// ---------------------------------------------------------------------------

/// Builds the request, sends it, and parses the response.
///
/// The `headers` closure receives a bare `RequestBuilder` (with Content-Type and User-Agent
/// already set) and must add the Authorization header (or any provider-specific headers).
/// Returns the full ChatMessage (content + tool_calls) for native function calling support.
async fn call_api(
    client: &Client,
    url: &str,
    headers: impl Fn(RequestBuilder) -> RequestBuilder,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    extra_body: Option<Value>,
) -> Result<ChatMessage, ApiError> {
    // Build JSON body directly from references — avoids cloning the entire message list
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": false,
    });
    if let Some(t) = tools {
        body["tools"] = serde_json::json!(t);
    }
    if let Some(tc) = tool_choice {
        body["tool_choice"] = serde_json::json!(tc);
    }
    if let Some(ref eb) = extra_body {
        if let Some(obj) = eb.as_object() {
            for (k, v) in obj {
                body[k] = v.clone();
            }
        }
    }

    let req = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "TerminalAI-Agent/0.1.0");
    let req = headers(req);

    let resp = req
        .json(&body)
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        // Read response body for actual error details
        let body_text = resp.text().await.unwrap_or_default();
        let detail = if body_text.is_empty() {
            http_status_detail(status).to_string()
        } else {
            // Try to extract error message from JSON body
            serde_json::from_str::<Value>(&body_text)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .or_else(|| v.get("message"))
                        .and_then(|e| e.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| {
                    // Fallback: use the raw body if short enough
                    if body_text.len() < 200 {
                        body_text
                    } else {
                        http_status_detail(status).to_string()
                    }
                })
        };
        return Err(ApiError::Http {
            status,
            detail,
        });
    }

    let chat: ChatResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::Parse(e.to_string()))?;

    chat.choices
        .into_iter()
        .next()
        .map(|c| c.message)
        .ok_or(ApiError::NoChoices)
}

/// Calls the OpenRouter API for a single model.
pub async fn call_openrouter(
    client: &Client,
    api_key: String,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    extra_body: Option<Value>,
) -> Result<ChatMessage, ApiError> {
    call_api(
        client,
        "https://openrouter.ai/api/v1/chat/completions",
        move |r| {
            r.header("Authorization", format!("Bearer {}", api_key))
                .header("HTTP-Referer", "https://github.com/terminal-ai-agent")
                .header("X-Title", "Terminal AI Agent")
        },
        model,
        messages,
        temperature,
        max_tokens,
        tools,
        tool_choice,
        extra_body,
    )
    .await
}

/// Calls the Groq API for a single model.
pub async fn call_groq(
    client: &Client,
    api_key: String,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    extra_body: Option<Value>,
) -> Result<ChatMessage, ApiError> {
    call_api(
        client,
        "https://api.groq.com/openai/v1/chat/completions",
        move |r| r.header("Authorization", format!("Bearer {}", api_key)),
        model,
        messages,
        temperature,
        max_tokens,
        tools,
        tool_choice,
        extra_body,
    )
    .await
}

/// Calls the NVIDIA NIM API for a single model.
///
/// Uses the same OpenAI-compatible endpoint as the Python agent.
pub async fn call_nvidia(
    client: &Client,
    api_key: String,
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    extra_body: Option<Value>,
) -> Result<ChatMessage, ApiError> {
    // Auto-add thinking params for nemotron models to enable chain-of-thought reasoning
    let eb = extra_body.or_else(|| {
        if model.contains("nemotron") || model.contains("nemotron-3") {
            Some(serde_json::json!({
                "chat_template_kwargs": {"enable_thinking": true},
                "reasoning_budget": 16384
            }))
        } else {
            None
        }
    });
    call_api(
        client,
        "https://integrate.api.nvidia.com/v1/chat/completions",
        move |r| r.header("Authorization", format!("Bearer {}", api_key)),
        model,
        messages,
        temperature,
        max_tokens,
        tools,
        tool_choice,
        eb,
    )
    .await
}

/// Calls the Google Gemini API (OpenAI-compatible endpoint) for a single model.
///
/// Uses `x-goog-api-key` header instead of Bearer auth – Google API keys
/// are not accepted as Bearer tokens.
// ---------------------------------------------------------------------------
// Conversation memory (RwLock + VecDeque for O(1) operations)
// ---------------------------------------------------------------------------

static CONVERSATION: LazyLock<RwLock<VecDeque<ChatMessage>>> =
    LazyLock::new(|| RwLock::new(VecDeque::new()));

/// Query counter for debouncing saves — only writes to disk every N queries.
static QUERY_COUNT: LazyLock<std::sync::atomic::AtomicU32> =
    LazyLock::new(|| std::sync::atomic::AtomicU32::new(0));

const MAX_TURNS: usize = 12;
const SAVE_EVERY: u32 = 5;

/// Simple in-memory response cache: query_hash -> response
static RESPONSE_CACHE: LazyLock<std::sync::Mutex<HashMap<u64, String>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
const CACHE_MAX_ENTRIES: usize = 64;

/// Background-generated follow-up suggestions for the current conversation
static FOLLOW_UP_SUGGESTIONS: LazyLock<RwLock<Vec<String>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));

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

/// Persists the current conversation history to disk (non-blocking, debounced).
///
/// Only writes every 5th query to avoid blocking the main thread on disk I/O.
pub async fn save_conversation() {
    use std::sync::atomic::Ordering;
    let count = QUERY_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count % SAVE_EVERY != 0 {
        return;
    }
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

/// Force-save conversation to disk (used on exit/shutdown).
pub async fn force_save_conversation() {
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
// Response cache helpers
// ---------------------------------------------------------------------------

fn cache_key(query: &str, temperature: f32) -> u64 {
    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    temperature.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn cache_get(query: &str, temperature: f32) -> Option<String> {
    let key = cache_key(query, temperature);
    let cache = RESPONSE_CACHE.lock().unwrap();
    cache.get(&key).cloned()
}

fn cache_put(query: &str, temperature: f32, response: &str) {
    let key = cache_key(query, temperature);
    let mut cache = RESPONSE_CACHE.lock().unwrap();
    if cache.len() >= CACHE_MAX_ENTRIES {
        cache.clear();
    }
    cache.insert(key, response.to_string());
}

pub async fn get_suggestions() -> Vec<String> {
    FOLLOW_UP_SUGGESTIONS.read().await.clone()
}

/// Attempts to generate follow-up questions using the fastest available model (Groq).
async fn generate_followups(client: &Client, query: &str, response: &str) -> Vec<String> {
    let prompt = format!(
        "Based on this Q&A:\nQ: {}\nA: {}\n\nSuggest 2-3 follow-up questions as a JSON array of strings. Return ONLY the array, no other text.",
        query, response
    );

    let msg = ChatMessage {
        role: "user".to_string(),
        content: Some(prompt),
        tool_calls: None,
        tool_call_id: None,
    };
    let msgs = vec![msg];

    let gk = get_groq_key();
    if let Ok(ref key) = gk {
        if let Ok(result) = call_groq(client, key.clone(), "llama-3.3-70b-versatile", &msgs, 0.5, 256, None, None, None).await {
            if let Some(content) = result.content {
                let parsed = parse_json_array(&content);
                if !parsed.is_empty() {
                    return parsed;
                }
            }
        }
    }
    Vec::new()
}

fn parse_json_array(s: &str) -> Vec<String> {
    let trimmed = s.trim();
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&trimmed[start..=end]) {
                return arr;
            }
        }
    }
    Vec::new()
}

/// Monotonic generation counter — prevents stale self-reflection from overwriting
/// a newer response when the user asks a follow-up before reflection finishes.
static GENERATION: LazyLock<std::sync::atomic::AtomicU64> =
    LazyLock::new(|| std::sync::atomic::AtomicU64::new(0));

/// Self-reflection: uses the fastest available model (Groq) to critique the response
/// and optionally produce an improved version. Runs in background after main display.
async fn self_reflect(client: &Client, query: &str, response: &str) -> Option<String> {
    let prompt = format!(
        "Review this response.\n\nQuery: {}\nResponse: {}\n\n\
        Is it accurate, complete, and well-structured? \
        If you can improve it, return ONLY the improved version. \
        If it's already good, return exactly: NO_CHANGE\n\n\
        Improved response:",
        query, response
    );

    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(prompt),
        tool_calls: None,
        tool_call_id: None,
    }];

    let gk = get_groq_key();
    let key = gk.as_ref().ok()?;
    let result = call_groq(client, key.clone(), "llama-3.3-70b-versatile", &msgs, 0.3, 1024, None, None, None).await.ok()?;
    let content = result.content?;
    let trimmed = content.trim().to_string();

    if !trimmed.is_empty()
        && !trimmed.eq_ignore_ascii_case("no_change")
        && trimmed.len() > response.len() / 3
    {
        let cleaned = trimmed
            .strip_prefix("Improved response:")
            .unwrap_or(&trimmed)
            .strip_prefix("improved response:")
            .unwrap_or(&trimmed)
            .trim()
            .to_string();
        return Some(cleaned);
    }
    None
}

/// Builds a lightweight project context summary from the current working directory.
/// Scans top-level files and key source directories to give the coding agent
/// awareness of the project structure. Returns a compact string suitable for
/// injection into the system prompt.
fn build_project_context() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut ctx = String::new();

    // Read project name from Cargo.toml or package.json
    let project_name = std::fs::read_to_string(cwd.join("Cargo.toml"))
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.trim().starts_with("name ="))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| v.trim().trim_matches('"').to_string())
        })
        .or_else(|| {
            std::fs::read_to_string(cwd.join("package.json")).ok().and_then(|s| {
                serde_json::from_str::<Value>(&s).ok()
                    .and_then(|v| v.get("name").and_then(|n| n.as_str().map(String::from)))
            })
        })
        .unwrap_or_default();

    if !project_name.is_empty() {
        ctx.push_str(&format!("Project: {}\n", project_name));
    }

    // Collect source files (respect .gitignore-like patterns)
    let mut files: Vec<String> = Vec::new();
    let ignore_dirs = [".git", "node_modules", "target", ".venv", "venv", "__pycache__", ".opencode"];

    if let Ok(entries) = std::fs::read_dir(&cwd) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if ignore_dirs.contains(&fname) || fname.starts_with('.') {
                    continue;
                }
                // Look for source files one level deep
                if let Ok(sub) = std::fs::read_dir(&path) {
                    for sub_entry in sub.flatten() {
                        let sp = sub_entry.path();
                        if sp.is_file() {
                            let ext = sp.extension().and_then(|e| e.to_str()).unwrap_or("");
                            let src_exts = ["rs", "py", "js", "ts", "go", "java", "rb", "c", "cpp", "h", "hpp", "toml", "json", "yaml", "yml", "md", "sh", "css", "html", "svelte", "vue"];
                            if src_exts.contains(&ext) && files.len() < 30 {
                                let rel = sp.strip_prefix(&cwd).unwrap_or(&sp).display().to_string();
                                let line_count = std::fs::read_to_string(&sp).ok()
                                    .map(|c| c.lines().count().to_string())
                                    .unwrap_or_default();
                                files.push(format!("  {} ({} lines)", rel, line_count));
                            }
                        }
                    }
                }
            } else if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                let top_exts = ["rs", "py", "js", "ts", "toml", "json", "yaml", "yml", "md", "sh", "dockerfile", "gitignore"];
                if top_exts.contains(&ext) || path.file_name().and_then(|n| n.to_str()) == Some("Dockerfile") {
                    let rel = path.strip_prefix(&cwd).unwrap_or(&path).display().to_string();
                    let line_count = std::fs::read_to_string(&path).ok()
                        .map(|c| c.lines().count().to_string())
                        .unwrap_or_default();
                    files.push(format!("  {} ({} lines)", rel, line_count));
                }
            }
        }
    }

    if !files.is_empty() {
        ctx.push_str("Files:\n");
        for f in files {
            ctx.push_str(&f);
            ctx.push('\n');
        }
    }

    if !ctx.is_empty() {
        ctx.insert_str(0, "[Project context]\n");
        ctx.push_str("[/Project context]");
    }
    ctx
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

fn strip_star_hash_re() -> &'static Regex {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[*#]").unwrap());
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

// ---------------------------------------------------------------------------
// Markdown table formatting
// ---------------------------------------------------------------------------

/// Detects if a line is a markdown table row (starts/ends with `|`, has at least 2 columns).
fn is_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed[1..].trim_end_matches('|').contains('|')
}

/// Detects if a line is a markdown table separator row (contains `---` between pipes).
fn is_table_separator(line: &str) -> bool {
    line.trim().starts_with('|') && line.contains("---")
}

/// Parses cells from a markdown table row, trimming whitespace.
fn parse_table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .map(|s| s.trim().to_string())
        .collect()
}

/// Formats a block of consecutive markdown table rows into a box-drawing table
/// with ANSI colors. Header row gets bold bright yellow, data rows get green.
/// Long cell content is word-wrapped to fit within column widths.
fn format_table_rows(rows: &[&str], term_w: usize) -> String {
    if rows.is_empty() {
        return String::new();
    }

    // Parse all rows, skipping the separator
    let mut parsed_rows: Vec<Vec<String>> = Vec::new();
    let mut has_header = false;

    for row in rows {
        let trimmed = row.trim();
        if is_table_separator(trimmed) {
            has_header = true;
            continue;
        }
        let cells = parse_table_cells(trimmed);
        if !cells.is_empty() {
            parsed_rows.push(cells);
        }
    }

    if parsed_rows.is_empty() {
        return String::new();
    }

    // Determine column count (max across all rows for ragged tables)
    let col_count = parsed_rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 {
        return String::new();
    }

    // Pad rows to equal column count
    for row in &mut parsed_rows {
        while row.len() < col_count {
            row.push(String::new());
        }
    }

    // Calculate column widths based on content
    let mut col_widths = vec![0usize; col_count];
    for row in &parsed_rows {
        for (i, cell) in row.iter().enumerate() {
            let visual_len = cell.chars().count();
            if visual_len > col_widths[i] {
                col_widths[i] = visual_len;
            }
        }
    }

    // Cap total width to terminal width with smarter proportional distribution
    let padding_total = col_count * 3 + 1;
    let total_content: usize = col_widths.iter().sum();
    let total_width = total_content + padding_total;
    if total_width > term_w && total_content > 0 {
        let available = term_w.saturating_sub(padding_total).max(col_count * 3);
        let ratio = available as f64 / total_content as f64;
        if ratio < 1.0 {
            // Wider columns get more space, but all get at least 5
            for w in &mut col_widths {
                *w = (*w as f64 * ratio).max(5.0) as usize;
            }
            // If still too wide after applying minimums, scale again
            let new_total: usize = col_widths.iter().sum();
            if new_total > available {
                let second_ratio = available as f64 / new_total as f64;
                for w in &mut col_widths {
                    *w = (*w as f64 * second_ratio).max(3.0) as usize;
                }
            }
        }
    }

    // Helper: render a single line of a cell with padding
    let format_cell_line = |line: &str, width: usize, _is_header: bool, _row_index: usize| -> String {
        let pad = width.saturating_sub(line.chars().count());
        format!(" {} {:pad$} ", line, "", pad = pad)
    };

    // Word-wrap each cell's content to its column width.
    // wrapped_lines[row_idx][cell_idx] = Vec<String> of wrapped lines (raw text)
    let mut wrapped_lines: Vec<Vec<Vec<String>>> = Vec::new();
    let mut row_heights: Vec<usize> = Vec::new();

    for row in &parsed_rows {
        let mut row_content: Vec<Vec<String>> = Vec::new();
        let mut max_lines_for_row = 1usize;

        for (i, cell) in row.iter().enumerate() {
            let col_w = col_widths[i];
            let lines: Vec<String> = if cell.chars().count() > col_w {
                wrap(cell, col_w)
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                vec![cell.clone()]
            };
            let n = lines.len();
            if n > max_lines_for_row {
                max_lines_for_row = n;
            }
            row_content.push(lines);
        }
        wrapped_lines.push(row_content);
        row_heights.push(max_lines_for_row);
    }

    // Helper: build an ASCII horizontal rule (accepts legacy args for compatibility)
    let hrule = |_: &str, _: &str, _: &str| -> String {
        let mut s = String::new();
        s.push('+');
        for (_i, w) in col_widths.iter().enumerate() {
            s.push_str(&"-".repeat(*w + 2));
            s.push('+');
        }
        s
    };

    let mut out = String::new();

    // Top border (ASCII)
    out.push_str(&format!("[36m{}[0m
", hrule("┌", "┬", "┐")));

    for (row_idx, _row) in parsed_rows.iter().enumerate() {
        let row_height = row_heights[row_idx];
        let is_header = has_header && row_idx == 0;

        // Render each text line of this row (multi-line cells span multiple lines)
        for line_idx in 0..row_height {
            out.push('|');
            for cell_idx in 0..col_count {
                let col_w = col_widths[cell_idx];
                let cell_lines = &wrapped_lines[row_idx][cell_idx];
                let cell_text = if line_idx < cell_lines.len() {
                    &cell_lines[line_idx]
                } else {
                    ""
                };
                let rendered = format_cell_line(cell_text, col_w, is_header, row_idx);
                out.push_str(&rendered);
                out.push('|');
            }
            out.push('\n');
        }

        // Grid separator after each row except the last
        if row_idx < parsed_rows.len() - 1 {
            out.push_str(&format!("[36m{}[0m
", hrule("├", "┼", "┤")));
        }
    }

    // Bottom border
    out.push_str(&format!("[36m{}[0m", hrule("└", "┴", "┘")));

    out
}


/// Renders a model response into a bordered string with ANSI color codes.
///
/// * Strips markdown formatting (`**bold**`, `### headings`, `` `inline code` ``, `* list`)
/// * Converts markdown tables into box-drawing tables with proper column alignment
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
    let strip_re = strip_star_hash_re();
    let mut in_code = false;
    // Buffer for table rows (markdown tables are consecutive |...| lines)
    let mut table_buf: Vec<&str> = Vec::new();

    /// Flush any buffered table rows into `lines` as individual padded lines.
    fn flush_table(table_buf: &mut Vec<&str>, lines: &mut Vec<String>, term_w: usize) {
        if table_buf.is_empty() {
            return;
        }
        let table = format_table_rows(table_buf, term_w);
        if !table.is_empty() {
            // Split multi-line table into individual lines so the final padding loop
            // handles each line correctly
            for tbl_line in table.lines() {
                lines.push(tbl_line.to_string());
            }
        }
        table_buf.clear();
    }

    for raw_line in resp.lines() {
        let trimmed = raw_line.trim();

        // Detect code fences
        if trimmed.starts_with("```") {
            flush_table(&mut table_buf, &mut lines, term_w);
            in_code = !in_code;
            continue;
        }

        if in_code {
            // Code blocks: keep as plain text (no colors)
            let wrapped: Vec<String> = wrap(raw_line, inner_w).into_iter().map(|s| s.to_string()).collect();
            lines.extend(wrapped);
            continue;
        }

        // Detect markdown table rows
        if is_table_row(trimmed) {
            table_buf.push(trimmed);
            continue;
        }

        // If we were building a table and this line isn't a table row, flush
        flush_table(&mut table_buf, &mut lines, term_w);

        let _is_heading = trimmed.starts_with("### ")
            || trimmed.starts_with("## ")
            || trimmed.starts_with("# ");
        let _is_bullet = trimmed.starts_with("- ") || trimmed.starts_with("* ");
        let _is_shell_cmd = sc_re.is_match(raw_line);
        let _is_acronym = ac_re.is_match(raw_line);

        let raw_stripped = h_re.replace_all(raw_line, "");
        let raw_stripped = list_star_re().replace_all(&raw_stripped, "");

        let wrapped: Vec<String> = wrap(&raw_stripped, inner_w)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        for sub in wrapped {
            let mut processed = sub;
            // Remove markdown formatting without colors
            processed = ic_re.replace_all(&processed, "$1").to_string();
            processed = b_re.replace_all(&processed, "$1").to_string();
            // Strip any remaining * or # characters
            processed = strip_re.replace_all(&processed, "").to_string();
            lines.push(processed);
        }
    }

    // Flush any remaining table buffer at end of input
    flush_table(&mut table_buf, &mut lines, term_w);

    if lines.is_empty() {
        lines.push(String::new());
    }

    // Simple horizontal line for top/bottom border (ASCII)
    let hline = "-".repeat(inner_w + 2);
    let mut out = String::new();
    out.push_str(&format!("{}\n", hline));
    for line in &lines {
        let stripped = a_re.replace_all(line, "").to_string();
        let pad = inner_w - stripped.chars().count().min(inner_w);
        out.push_str(&format!("{} {:pad$}\n", line, "", pad = pad));
    }
    out.push_str(&hline);
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
    let query_lower = query.to_ascii_lowercase();
    
    // Check for factual keywords - case-insensitive without additional allocation
    if query_lower.contains("what is") || query_lower.contains("who is") || 
       query_lower.contains("when did") || query_lower.contains("where is") || 
       query_lower.contains("how many") || query_lower.contains("define") ||
       query_lower.contains("explain") || query_lower.contains("difference between") ||
       query_lower.contains("compare") || query_lower.contains("version") ||
       query_lower.contains("release date") || query_lower.contains("population") ||
       query_lower.contains("capital") {
        return 0.2;
    }
    
    // Check for code keywords
    if query_lower.contains("code") || query_lower.contains("function") ||
       query_lower.contains("implement") || query_lower.contains("refactor") ||
       query_lower.contains("debug") || query_lower.contains("error") ||
       query_lower.contains("bug") || query_lower.contains("fix") ||
       query_lower.contains("algorithm") || query_lower.contains("compile") ||
       query_lower.contains("rust") || query_lower.contains("python") ||
       query_lower.contains("javascript") || query_lower.contains("dockerfile") ||
       query_lower.contains("git") {
        return 0.4;
    }
    
    // Check for creative keywords
    if query_lower.contains("write") || query_lower.contains("create") ||
       query_lower.contains("generate") || query_lower.contains("design") ||
       query_lower.contains("imagine") || query_lower.contains("poem") ||
       query_lower.contains("story") || query_lower.contains("haiku") ||
       query_lower.contains("song") || query_lower.contains("brainstorm") {
        return 0.8;
    }
    
    0.5
}

/// Returns a default max_tokens value based on query complexity.
fn auto_max_tokens(query: &str) -> u32 {
    let word_count = query.split_whitespace().count();
    if word_count > 50 {
        4096
    } else if query.len() > 200 {
        4096
    } else {
        2048
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

    // Structure score: headings, lists, code blocks, bold text, tables
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
    // Bonus for markdown tables — detect the separator row (| --- |) which is
    // a strong signal the model used structured comparison formatting
    if resp.lines().any(|l| l.trim().starts_with('|') && l.contains("---")) {
        structure += 15.0;
    }
    if lines > 3.0 {
        structure += 5.0;
    }

    // Penalty for genuine refusals / low-effort
    // NOTE: "as an ai" and "i don't have" / "i do not have" are NOT refusals —
    // they are common conversational fillers that often precede valid answers.
    let lower = resp.to_lowercase();
    let refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "sorry, i", "unfortunately, i",
    ];
    let refusal_penalty: f64 = refusal_phrases
        .iter()
        .filter(|p| lower.contains(*p))
        .map(|_| 15.0)
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
    let mut blank_count = 0u32;

    for line in resp.lines() {
        let trimmed = line.trim();
        
        // Skip if empty
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                result.push('\n');
            }
            continue;
        }
        
        blank_count = 0;
        
        // Check prefixes without allocation - use case-insensitive comparison
        let processed_line = {
            let lower_trim = trimmed.to_ascii_lowercase();
            if lower_trim.starts_with("here is ") {
                &trimmed[8..].trim_start()
            } else if lower_trim.starts_with("here are ") {
                &trimmed[9..].trim_start()
            } else if lower_trim.starts_with("sure, ") {
                &trimmed[6..].trim_start()
            } else if lower_trim.starts_with("of course, ") {
                &trimmed[11..].trim_start()
            } else if lower_trim.starts_with("certainly, ") {
                &trimmed[11..].trim_start()
            } else {
                trimmed
            }
        };
        
        result.push_str(processed_line);
        result.push('\n');
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
            // Keep first 100 chars of each old message as summary (char-safe, no panic on multi-byte)
            let char_count = trimmed.chars().count();
            if char_count > 100 {
                let truncated: String = trimmed.chars().take(100).collect();
                summary_parts.push(format!("{}...", truncated));
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

/// Cached system prompt — built once, reused on every query.
fn chat_system_prompt() -> &'static str {
    static PROMPT: LazyLock<String> = LazyLock::new(|| {
        r"You are a highly capable AI assistant. Follow these rules strictly:

## Reasoning (CRITICAL)
- For complex questions, reason step-by-step internally before answering
- Break down multi-part questions into sub-problems and address each one
- Consider multiple perspectives and edge cases
- If you're uncertain, acknowledge the uncertainty and explain why
- Always verify your reasoning — check for contradictions or leaps in logic

## Response quality
- Lead with the direct answer, then explain if needed
- Use concrete examples, numbers, and specific facts — avoid vague generalities
- For technical topics, include code examples where relevant
- Keep responses focused — answer what was asked, nothing extra

## Formatting
- Use ### headings for major sections
- Use **bold** for key terms on first mention
- Use backtick-inline-code for commands, file paths, flags, and technical identifiers
- Use triple-backtick code blocks for multi-line code, configs, or shell commands
- Use - bullet points for lists of 3+ items
- Use numbered lists for sequential steps

## Tables (CRITICAL — used for all comparisons, tabular data, and structured points)
- ALL comparisons, feature comparisons, pricing, specifications, pros/cons, metrics, or any multi-column data MUST use markdown table format with pipes:
  | Header 1 | Header 2 | Header 3 |
  |----------|----------|----------|
  | Cell 1   | Cell 2   | Cell 3   |
- This includes: comparing tools, products, versions, languages, frameworks, providers, plans, features, benchmarks, and any structured points
- Do NOT use loose column alignment or plain text lists for comparison data — always use a proper markdown table
- Tables are rendered as beautiful box-drawing grids with borders on all sides

## Personality
- Be direct and confident — don't hedge with hedging phrases
- Vary your examples and analogies each time
- Match the depth to the question: simple question = simple answer
- If the question is ambiguous, pick the most useful interpretation and answer it
- Anticipate follow-up questions and address them preemptively

## What NOT to do
- Don't start with filler phrases like Here is or Sure — just answer
- Don't repeat the question back
- Don't apologize or give disclaimers unless truly necessary
- Don't include unnecessary preamble or closing remarks"
            .to_string()
    });
    &PROMPT
}

/// Per-model rate-limit cooldown: tracks when a model was last rate-limited
/// so we skip it until the cooldown expires.
static RATE_LIMIT_COOLDOWNS: LazyLock<std::sync::Mutex<HashMap<String, Instant>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

const RATE_LIMIT_COOLDOWN_SECS: u64 = 5;

/// Marks `model` as rate-limited for `RATE_LIMIT_COOLDOWN_SECS`.
fn mark_model_rate_limited(model: &str) {
    let mut map = RATE_LIMIT_COOLDOWNS.lock().unwrap();
    // Clean stale entries while we're at it
    let now = Instant::now();
    map.retain(|_, v| *v > now);
    map.insert(model.to_string(), now + Duration::from_secs(RATE_LIMIT_COOLDOWN_SECS));
}

/// Simple jitter: returns `base_ms` ± up to 25% using system-time nanosecond bits
/// so retries don't align (avoiding thundering-herd).
fn jitter_ms(base_ms: u64) -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let divisor = (base_ms as u32 / 2).max(1);
    let offset = (nanos % divisor) as u64;
    if nanos % 2 == 0 {
        base_ms + offset
    } else {
        base_ms.saturating_sub(offset)
    }
}

/// Helper to try a single model call with retry for 429s.
///
/// Uses exponential backoff with jitter and a global cooldown tracker.
/// On 429, backs off then tries different models instead of exhausting retries on one.
pub(crate) async fn try_model<F, Fut>(f: F, model: &str, delay_ms: u64, timeout_secs: u64) -> (Option<ChatMessage>, String)
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<ChatMessage, ApiError>>,
{
    // 429 retries: exponential backoff with jitter.
    const MAX_429_RETRIES: u32 = 3;

    let mut last_err = String::new();
    let mut consecutive_429s = 0u32;
    for _ in 0..(MAX_429_RETRIES + 1) {
        let res = timeout(Duration::from_secs(timeout_secs), f()).await;
        match res {
            Ok(Ok(msg)) => {
                return (Some(msg), String::new());
            }
            Ok(Err(ApiError::Http { status: 429, .. })) => {
                consecutive_429s += 1;
                last_err = format!("{}: Rate limited (attempt {})", model, consecutive_429s);
                if consecutive_429s >= MAX_429_RETRIES {
                    mark_model_rate_limited(model);
                    last_err = format!("{}: Rate limited — cooldown {}s", model, RATE_LIMIT_COOLDOWN_SECS);
                    break;
                }
                let backoff = jitter_ms(delay_ms * (1 << (consecutive_429s - 1)));
                tokio::time::sleep(Duration::from_millis(backoff)).await;
            }
            Ok(Err(e)) => {
                last_err = format!("{}: {}", model, e);
                break;
            }
            Err(_) => {
                last_err = format!("{}: Timeout ({}s)", model, timeout_secs);
                break;
            }
        }
    }
    (None, last_err)
}

/// Streams a response from the fastest available model (Groq), printing tokens
/// as they arrive for a ChatGPT-like real-time experience.
/// Falls back gracefully to the existing parallel racing on any error.
async fn try_stream_response(
    client: &Client,
    _query: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Option<String> {
    let gk = get_groq_key();
    let key = gk.as_ref().ok()?;

    let body = serde_json::json!({
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": true,
    });

    let req = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("User-Agent", "TerminalAI-Agent/0.1.0")
        .header("Authorization", format!("Bearer {}", key))
        .json(&body);

    let resp = match req.send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return None,
    };

    // Show a subtle waiting indicator (overwritten by first token)
    print!("{}", "   ".dimmed());
    let _ = std::io::stdout().flush();

    let mut full_content = String::new();
    let mut stream = resp.bytes_stream();
    let mut first_token = true;

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => break,
        };
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                    if let Some(choice) = choices.first() {
                        if let Some(delta) = choice.get("delta") {
                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                if first_token {
                                    // Clear the waiting indicator
                                    first_token = false;
                                }
                                full_content.push_str(content);
                                print!("{}", content);
                                std::io::stdout().flush().ok();
                            }
                        }
                    }
                }
            }
        }
    }

    if full_content.is_empty() {
        return None;
    }

    println!();
    // Print horizontal rule to match format_response style
    let term_w = safe_termwidth();
    let inner_w = term_w.saturating_sub(2).max(20);
    println!("{}", "-".repeat(inner_w + 2));

    Some(post_process_response(&full_content))
}

/// Streams a response from NVIDIA nemotron with thinking support, printing
/// reasoning tokens (dimmed) and content tokens as they arrive.
/// Falls back gracefully to Groq streaming or parallel racing on any error.
async fn try_stream_nvidia(
    client: &Client,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: u32,
) -> Option<String> {
    let nk = get_nvidia_key();
    if nk.is_empty() {
        return None;
    }

    let body = serde_json::json!({
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": true,
        "chat_template_kwargs": {"enable_thinking": true},
        "reasoning_budget": 16384,
    });

    let req = client
        .post("https://integrate.api.nvidia.com/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("User-Agent", "TerminalAI-Agent/0.1.0")
        .header("Authorization", format!("Bearer {}", nk))
        .json(&body);

    let resp = match req.send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return None,
    };

    // Show a subtle waiting indicator (overwritten by first token)
    print!("{}", "   ".dimmed());
    let _ = std::io::stdout().flush();

    let mut full_content = String::new();
    let mut stream = resp.bytes_stream();
    let mut first_token = true;
    let mut in_thinking = false;

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => break,
        };
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                    if let Some(choice) = choices.first() {
                        if let Some(delta) = choice.get("delta") {
                            // Print reasoning content (chain-of-thought) in dimmed style
                            if let Some(rc) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                                if !rc.is_empty() {
                                    if first_token {
                                        first_token = false;
                                    }
                                    if !in_thinking {
                                        in_thinking = true;
                                    }
                                    print!("{}", rc.dimmed());
                                    std::io::stdout().flush().ok();
                                }
                            }
                            // Print actual response content normally
                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                if first_token {
                                    first_token = false;
                                }
                                // If we were in thinking mode and now have content, print separator
                                if in_thinking {
                                    in_thinking = false;
                                    // Print a brief separator between thinking and content
                                }
                                full_content.push_str(content);
                                print!("{}", content);
                                std::io::stdout().flush().ok();
                            }
                        }
                    }
                }
            }
        }
    }

    if full_content.is_empty() {
        return None;
    }

    println!();
    // Print horizontal rule to match format_response style
    let term_w = safe_termwidth();
    let inner_w = term_w.saturating_sub(2).max(20);
    println!("{}", "-".repeat(inner_w + 2));

    Some(post_process_response(&full_content))
}

/// Runs a user query against models concurrently with scoring and fallback.
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

    // Check cache before making API calls
    if let Some(cached) = cache_get(query, effective_temp) {
        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: Some(cached.clone()),
            tool_calls: None,
            tool_call_id: None,
        };
        push_conversation(assistant).await;
        save_conversation().await;
        println!("{}", format_response(&cached));
        return;
    }

    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: Some(query.to_string()),
        tool_calls: None,
        tool_call_id: None,
    };
    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: Some(chat_system_prompt().to_string()),
        tool_calls: None,
        tool_call_id: None,
    };

    let raw_history = conversation_history().await;
    // Compress old messages: keep only the last 4 in full, summarize the rest.
    // This keeps context small even after many conversation turns.
    let history = summarize_old_context(&raw_history, 4);

    let mut msg_vec = vec![system_msg];
    msg_vec.extend(history);
    // CRITICAL: include the current user query in the messages sent to the API!
    msg_vec.push(user_msg.clone());
    // Share via Arc so each model avoids cloning the entire message vector
    let msg_vec = Arc::new(msg_vec);

    // Collect API keys once
    let ak = get_openrouter_key();
    let gk = get_groq_key();
    let gk_val = gk.clone().unwrap_or_default();
    let nk = get_nvidia_key();
    let nk_qwen = get_nvidia_qwen_key();
    let no_keys = ak.is_empty() && gk.is_err() && nk.is_empty() && nk_qwen.is_empty();

    push_conversation(user_msg.clone()).await;

    // Try NVIDIA nemotron streaming first (strongest model with thinking support)
    // 120s timeout; falls back to Groq streaming then parallel racing
    if !nk.is_empty() {
        if let Some(streamed) = timeout(Duration::from_secs(120), try_stream_nvidia(client, &msg_vec, effective_temp, max_tokens)).await.ok().flatten() {
            cache_put(query, effective_temp, &streamed);
            let assistant = ChatMessage {
                role: "assistant".to_string(),
                content: Some(streamed.clone()),
                tool_calls: None,
                tool_call_id: None,
            };
            push_conversation(assistant).await;
            save_conversation().await;

            // Background follow-ups
            let fc = client.clone();
            let fq = query.to_string();
            let fr = streamed.clone();
            tokio::spawn(async move {
                let suggestions = generate_followups(&fc, &fq, &fr).await;
                *FOLLOW_UP_SUGGESTIONS.write().await = suggestions;
            });

            // Background self-reflection
            let sr_client = client.clone();
            let sr_query = query.to_string();
            let sr_response = streamed.clone();
            let sr_gen = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            tokio::spawn(async move {
                if let Some(improved) = self_reflect(&sr_client, &sr_query, &sr_response).await {
                    let current_gen = GENERATION.load(std::sync::atomic::Ordering::Relaxed);
                    if current_gen == sr_gen {
                        let mut conv = CONVERSATION.write().await;
                        if let Some(last) = conv.back_mut() {
                            if last.role == "assistant" && last.content.as_deref() == Some(&sr_response) {
                                last.content = Some(improved);
                            }
                        }
                    }
                }
            });
            return;
        }
    }

    // Try Groq streaming next (fast fallback)
    // 120s timeout; falls back to parallel racing on failure
    if let Some(streamed) = timeout(Duration::from_secs(120), try_stream_response(client, query, &msg_vec, effective_temp, max_tokens)).await.ok().flatten() {
        cache_put(query, effective_temp, &streamed);
        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: Some(streamed.clone()),
            tool_calls: None,
            tool_call_id: None,
        };
        push_conversation(assistant).await;
        save_conversation().await;

        // Background follow-ups
        let fc = client.clone();
        let fq = query.to_string();
        let fr = streamed.clone();
        tokio::spawn(async move {
            let suggestions = generate_followups(&fc, &fq, &fr).await;
            *FOLLOW_UP_SUGGESTIONS.write().await = suggestions;
        });

        // Background self-reflection
        let sr_client = client.clone();
        let sr_query = query.to_string();
        let sr_response = streamed.clone();
        let sr_gen = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        tokio::spawn(async move {
            if let Some(improved) = self_reflect(&sr_client, &sr_query, &sr_response).await {
                let current_gen = GENERATION.load(std::sync::atomic::Ordering::Relaxed);
                if current_gen == sr_gen {
                    let mut conv = CONVERSATION.write().await;
                    if let Some(last) = conv.back_mut() {
                        if last.role == "assistant" && last.content.as_deref() == Some(&sr_response) {
                            last.content = Some(improved);
                        }
                    }
                }
            }
        });
        return;
    }

    let mut attempts: Vec<String> = Vec::new();

    // Run ALL models concurrently (FuturesUnordered) — return the FIRST valid response
    // Groq is fastest (~1-3s) and skips quality scoring since llama-3.3-70b is production-grade.
    // OpenRouter models run in parallel as fallback if Groq errors out.
    let mut response: Option<String> = None;
    {
        use futures_util::stream::FuturesUnordered;
        use futures_util::StreamExt;
        // Type-erase via Pin<Box<dyn Future>> so different async blocks are compatible
        let mut futs: FuturesUnordered<std::pin::Pin<Box<dyn std::future::Future<Output = (Option<String>, String)> + Send>>> = FuturesUnordered::new();
        // Add NVIDIA NIM models (production-grade, no rate limits — same as Python agent)
        if !nk.is_empty() {
            for model in NVIDIA_MODELS {
                let model_s = model.to_string();
                let nk_c = nk.clone();
                let mv = Arc::clone(&msg_vec);
                futs.push(Box::pin(async move {
                    let (msg, err) = try_model(
                        || call_nvidia(client, nk_c.clone(), &model_s, &mv, effective_temp, max_tokens, None, None, None),
                        &model_s,
                        1000,
                        12,  // NVIDIA NIM is slower but reliable, give it 12s
                    ).await;
                    if let Some(m) = msg {
                        let text = m.content.unwrap_or_default();
                        if !text.trim().is_empty() {
                            let processed = post_process_response(&text);
                            return (Some(processed), String::new());
                        }
                    }
                    (None, err)
                }));
            }
        }
        // Add NVIDIA Qwen model (dedicated API key)
        if !nk_qwen.is_empty() {
            for model in NVIDIA_QWEN_MODELS {
                let model_s = model.to_string();
                let nk_qwen_c = nk_qwen.clone();
                let mv = Arc::clone(&msg_vec);
                futs.push(Box::pin(async move {
                    let (msg, err) = try_model(
                        || call_nvidia(client, nk_qwen_c.clone(), &model_s, &mv, effective_temp, max_tokens, None, None, None),
                        &model_s,
                        1000,
                        12,
                    ).await;
                    if let Some(m) = msg {
                        let text = m.content.unwrap_or_default();
                        if !text.trim().is_empty() {
                            let processed = post_process_response(&text);
                            return (Some(processed), String::new());
                        }
                    }
                    (None, err)
                }));
            }
        }
        // Add Groq model
        if !gk_val.is_empty() {
            for model in GROQ_MODELS {
                let model_s = model.to_string();
                let gk = gk_val.clone();
                let mv = Arc::clone(&msg_vec);
                futs.push(Box::pin(async move {
                    let (msg, err) = try_model(
                        || call_groq(client, gk.clone(), &model_s, &mv, effective_temp, max_tokens, None, None, None),
                        &model_s,
                        1000,
                        4,  // Groq is fast, timeout rapidly
                    ).await;
                    if let Some(m) = msg {
                        let text = m.content.unwrap_or_default();
                        if !text.trim().is_empty() {
                            let processed = post_process_response(&text);
                            return (Some(processed), String::new());
                        }
                    }
                    (None, err)
                }));
            }
        }
        // Add all OpenRouter models
        if !ak.is_empty() {
            let or_models = get_models();
            for model in or_models {
                let model_s = model.clone();
                let ak_c = ak.clone();
                let mv = Arc::clone(&msg_vec);
                futs.push(Box::pin(async move {
                    let (msg, err) = try_model(
                        || call_openrouter(client, ak_c.clone(), &model_s, &mv, effective_temp, max_tokens, None, None, None),
                        &model_s,
                        1000,
                        4,  // OpenRouter free models are slow/unreliable, timeout rapidly
                    ).await;
                    if let Some(m) = msg {
                        let text = m.content.unwrap_or_default();
                        let processed = post_process_response(&text);
                        let score = score_response(&processed);
                        if score > 0.0 {
                            return (Some(processed), String::new());
                        }
                        return (None, format!("{}: low quality (score {:.0})", model_s, score));
                    }
                    (None, err)
                }));
            }
        }
        while let Some((resp_opt, err_str)) = futs.next().await {
            if let Some(r) = resp_opt {
                response = Some(r);
                break;
            }
            if !err_str.is_empty() {
                attempts.push(err_str);
            }
        }
    }

    if let Some(best) = response {
        cache_put(query, effective_temp, &best);

        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: Some(best.clone()),
            tool_calls: None,
            tool_call_id: None,
        };
        push_conversation(assistant).await;
        save_conversation().await;
        println!("{}", format_response(&best));

        // Spawn background follow-up generation (no impact on response speed)
        let fc = client.clone();
        let fq = query.to_string();
        let fr = best.clone();
        tokio::spawn(async move {
            let suggestions = generate_followups(&fc, &fq, &fr).await;
            *FOLLOW_UP_SUGGESTIONS.write().await = suggestions;
        });

        // Self-reflection: critique and improve response in background
        let sr_client = client.clone();
        let sr_query = query.to_string();
        let sr_response = best.clone();
        let sr_gen = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        tokio::spawn(async move {
            if let Some(improved) = self_reflect(&sr_client, &sr_query, &sr_response).await {
                let current_gen = GENERATION.load(std::sync::atomic::Ordering::Relaxed);
                if current_gen == sr_gen {
                    let mut conv = CONVERSATION.write().await;
                    if let Some(last) = conv.back_mut() {
                        if last.role == "assistant" && last.content.as_deref() == Some(&sr_response) {
                            last.content = Some(improved);
                        }
                    }
                }
            }
        });

        return;
    }

    CONVERSATION.write().await.pop_back();

    eprintln!("{}", "All models failed.".red());
    for a in &attempts {
        eprintln!("  {} {}", "•".yellow(), a.cyan());
    }
    if no_keys {
        eprintln!(
            "{}",
            "No API keys set. Export NVIDIA_API_KEY, NVIDIA_QWEN_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY."
                .yellow()
        );
    }
}

/// Enhanced system prompt for coding agent mode — detailed, structured, with error recovery.
fn coding_system_prompt() -> String {
    r#"You are an expert coding agent. You can read, write, edit files, run shell commands, search code, and find files.

## Reasoning (CRITICAL — follow this every time)
Before each action, think step-by-step:
1. What is the actual goal? Break it down
2. What do I know? What information do I still need?
3. What is the simplest approach that could work?
4. What could go wrong? How will I handle it?
5. After each action, verify the result before proceeding

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

## Planning phase (CRITICAL — output this first)
When given a multi-step task, FIRST output a numbered plan inside <plan> tags:
<plan>
  Step 1: [tool] what to do
  Step 2: [tool] what to do next
  ...
</plan>
Then execute each step strictly in order. Do NOT skip ahead.

## CRITICAL RULE — Write ALL files FIRST
Complete ALL write_file operations BEFORE running any bash commands.
Do NOT run bash commands that depend on files until after those files are created.
This means: first create every file, THEN run docker/make/test commands.

## Workflow
1. Understand the task — read relevant files first if needed
2. Plan your approach — list ALL files to create and ALL steps in <plan> tags
3. Write ALL files using write_file FIRST (one at a time)
4. Only AFTER all files are written, run bash commands (docker, make, etc.)
5. Verify your work — check files after editing, run tests after changes
6. If a tool call fails, analyze the error and try a different approach
7. When done, provide a clear summary of what was done

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
/// Uses native function calling API for better reliability, falls back to
/// text-based tool call parsing when native function calling is not supported.
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
    const MAX_MESSAGES: usize = 50;

    // Cache API keys and system prompt outside the loop
    let ak = get_openrouter_key();
    let gk = get_groq_key();
    let gk_val = gk.clone().unwrap_or_default();
    let nk = get_nvidia_key();
    let nk_qwen = get_nvidia_qwen_key();
    let system_prompt = coding_system_prompt();

    let system_msg = ChatMessage {
        role: "system".to_string(),
        content: Some(system_prompt),
        tool_calls: None,
        tool_call_id: None,
    };
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: Some(query.to_string()),
        tool_calls: None,
        tool_call_id: None,
    };

    // Inject project context as a system message for project awareness
    let project_ctx = build_project_context();
    let mut messages = if project_ctx.is_empty() {
        vec![system_msg, user_msg]
    } else {
        let ctx_msg = ChatMessage {
            role: "system".to_string(),
            content: Some(project_ctx),
            tool_calls: None,
            tool_call_id: None,
        };
        vec![system_msg, ctx_msg, user_msg]
    };

    let tc_re = tool_call_re();
    let mut iterations = 0;
    const MAX_ITER: usize = 40;

    // Track executed tool calls to detect and block duplicate loops
    let mut executed_calls: Vec<(String, String)> = Vec::new();

    // Build tool definitions for native function calling (done once, reused)
    let tool_defs: Vec<Tool> = tools::all_tool_defs()
        .into_iter()
        .filter_map(|v| {
            let name = v["function"]["name"].as_str()?.to_string();
            let description = v["function"]["description"].as_str()?.to_string();
            let parameters = v["function"]["parameters"].clone();
            Some(Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name,
                    description,
                    parameters,
                },
            })
        })
        .collect();
    let tools = Some(tool_defs);
    let tool_choice = Some(ToolChoice::Auto);

    fn trim_messages(msgs: &mut Vec<ChatMessage>) {
        if msgs.len() > MAX_MESSAGES {
            let system_msg = msgs[0].clone();
            // Try to preserve project context message (index 1) if present
            let ctx_msg = if msgs.len() > 2 && msgs[1].role == "system" && msgs[1].content.as_deref().map_or(false, |c| c.starts_with("[Project context]")) {
                Some(msgs[1].clone())
            } else {
                None
            };
            let ctx_offset = if ctx_msg.is_some() { 1 } else { 0 };
            let keep = msgs.len() - MAX_MESSAGES + 1 + ctx_offset;
            let mut new_msgs = vec![system_msg];
            if let Some(ctx) = ctx_msg {
                new_msgs.push(ctx);
            }
            new_msgs.extend_from_slice(&msgs[keep..]);
            // Shorten long tool result messages to preserve context
            for msg in &mut new_msgs {
                if msg.role == "tool" {
                    if let Some(ref content) = msg.content {
                        if content.len() > 300 {
                            let trimmed: String = content.chars().take(300).collect();
                            msg.content = Some(format!("{}... (truncated)", trimmed));
                        }
                    }
                }
            }
            *msgs = new_msgs;
        }
    }

    fn msg_to_tool_call(msg: &ChatMessage) -> Option<ToolCallText> {
        if let Some(ref tcs) = msg.tool_calls {
            let tc = tcs.first()?;
            let args: Value = serde_json::from_str(&tc.function.arguments).ok()?;
            return Some(ToolCallText {
                name: tc.function.name.clone(),
                arguments: args,
            });
        }
        None
    }

    loop {
        iterations += 1;
        if iterations > MAX_ITER {
            eprintln!("{}", "[code-agent] Max iterations reached.".red());
            break;
        }

        let mut attempts: Vec<String> = Vec::new();

        // Try NVIDIA, NVIDIA Qwen, Groq, and first OpenRouter model CONCURRENTLY
        let nv_fut = async {
            if nk.is_empty() {
                return (None::<ChatMessage>, Vec::new());
            }
            let mut att = Vec::new();
            for model in NVIDIA_MODELS {
                let (msg, err) = try_model(
                    || call_nvidia(client, nk.clone(), model, &messages, effective_temp, max_tokens, tools.clone(), tool_choice.clone(), None),
                    model, 1000, 12,
                ).await;
                if let Some(m) = msg {
                    if m.tool_calls.is_some() || tc_re.is_match(&m.content.as_deref().unwrap_or("")) {
                        return (Some(m), att);
                    }
                    let text = m.content.as_deref().unwrap_or("");
                    let score = score_response(text);
                    if score > 0.0 {
                        return (Some(m), att);
                    }
                    att.push(format!("{}: low quality (score {:.0})", model, score));
                } else {
                    att.push(err);
                }
            }
            (None::<ChatMessage>, att)
        };
        let groq_fut = async {
            if gk_val.is_empty() {
                return (None::<ChatMessage>, Vec::new());
            }
            let mut att = Vec::new();
            for model in GROQ_MODELS {
                let (msg, err) = try_model(
                    || call_groq(client, gk_val.clone(), model, &messages, effective_temp, max_tokens, tools.clone(), tool_choice.clone(), None),
                    model, 1000, 4,
                ).await;
                if let Some(m) = msg {
                    if m.tool_calls.is_some() || tc_re.is_match(&m.content.as_deref().unwrap_or("")) {
                        return (Some(m), att);
                    }
                    let text = m.content.as_deref().unwrap_or("");
                    let score = score_response(text);
                    if score > 0.0 {
                        return (Some(m), att);
                    }
                    att.push(format!("{}: low quality (score {:.0})", model, score));
                } else {
                    att.push(err);
                }
            }
            (None::<ChatMessage>, att)
        };
        let or_fut = async {
            if ak.is_empty() {
                return (None::<ChatMessage>, Vec::new());
            }
            let models = get_models();
            if models.is_empty() {
                return (None, Vec::new());
            }
            let mut att = Vec::new();
            let (msg, err) = try_model(
                || call_openrouter(client, ak.clone(), &models[0], &messages, effective_temp, max_tokens, tools.clone(), tool_choice.clone(), None),
                &models[0], 1000, 4,
            ).await;
            if let Some(m) = msg {
                if m.tool_calls.is_some() || tc_re.is_match(&m.content.as_deref().unwrap_or("")) {
                    return (Some(m), att);
                }
                let text = m.content.as_deref().unwrap_or("");
                let score = score_response(text);
                if score > 0.0 {
                    return (Some(m), att);
                }
                att.push(format!("{}: low quality (score {:.0})", models[0], score));
            } else {
                att.push(err);
            }
            (None::<ChatMessage>, att)
        };
        // Add NVIDIA Qwen parallel future
        let qwen_fut = async {
            if nk_qwen.is_empty() {
                return (None::<ChatMessage>, Vec::new());
            }
            let mut att = Vec::new();
            for model in NVIDIA_QWEN_MODELS {
                let (msg, err) = try_model(
                    || call_nvidia(client, nk_qwen.clone(), model, &messages, effective_temp, max_tokens, tools.clone(), tool_choice.clone(), None),
                    model, 1000, 12,
                ).await;
                if let Some(m) = msg {
                    if m.tool_calls.is_some() || tc_re.is_match(&m.content.as_deref().unwrap_or("")) {
                        return (Some(m), att);
                    }
                    let text = m.content.as_deref().unwrap_or("");
                    let score = score_response(text);
                    if score > 0.0 {
                        return (Some(m), att);
                    }
                    att.push(format!("{}: low quality (score {:.0})", model, score));
                } else {
                    att.push(err);
                }
            }
            (None::<ChatMessage>, att)
        };

        let ((nv_resp, nv_att), (groq_resp, groq_att), (or_resp, or_att)) = tokio::join!(nv_fut, groq_fut, or_fut);
        let (qwen_resp, qwen_att) = qwen_fut.await;
        attempts.extend(nv_att);
        attempts.extend(groq_att);
        attempts.extend(or_att);
        attempts.extend(qwen_att);

        // Pick NVIDIA response first (production-grade, no rate limits), then Qwen, then Groq
        let mut response_msg = nv_resp.or(qwen_resp).or(groq_resp).or(or_resp);

        // Try remaining OpenRouter models CONCURRENTLY (join_all)
        if response_msg.is_none() && !ak.is_empty() {
            let models = get_models();
            let remaining = &models[1..];
            if !remaining.is_empty() {
                use futures_util::future::join_all;
                let futs: Vec<_> = remaining.iter().map(|model| {
                    try_model(
                        || call_openrouter(client, ak.clone(), model, &messages, effective_temp, max_tokens, tools.clone(), tool_choice.clone(), None),
                        model,
                        1000,
                        4,
                    )
                }).collect();
                let results = join_all(futs).await;
                for (i, (msg, err)) in results.into_iter().enumerate() {
                    if let Some(m) = msg {
                        if m.tool_calls.is_some() || tc_re.is_match(&m.content.as_deref().unwrap_or("")) {
                            response_msg = Some(m);
                            break;
                        }
                        let text = m.content.as_deref().unwrap_or("");
                        let score = score_response(text);
                        if score > 0.0 {
                            response_msg = Some(m);
                            break;
                        }
                        attempts.push(format!("{}: low quality (score {:.0})", remaining[i], score));
                    } else {
                        attempts.push(err);
                    }
                }
            }
        }

        let msg = match response_msg {
            Some(m) => m,
            None => {
                eprintln!("{}", "[code-agent] All models failed.".red());
                for a in &attempts {
                    eprintln!("  {} {}", "•".yellow(), a.cyan());
                }
                break;
            }
        };

        let response_text = msg.content.as_deref().unwrap_or("").to_string();

        // Detect and skip duplicate tool calls (model looping on same action)
        let tc_key = if let Some(tc) = msg_to_tool_call(&msg) {
            Some((tc.name.clone(), tc.arguments.to_string()))
        } else {
            tc_re.captures(&response_text).and_then(|caps| {
                let tag = caps.get(1)?.as_str().to_string();
                let after_tag = &response_text[caps.get(0).unwrap().end()..];
                let json_str = extract_balanced_json(after_tag)?;
                Some((tag, json_str.to_string()))
            })
        };

        if let Some(ref key) = tc_key {
            if executed_calls.iter().any(|(n, a)| n == &key.0 && a == &key.1) {
                let dup_msg = format!("⚠️ Duplicate tool call '{}' with same arguments — skipped to prevent loop", key.0);
                eprintln!("{}", dup_msg.yellow());
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(response_text),
                    tool_calls: None,
                    tool_call_id: None,
                });
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some(format!("[system] {} Please try a different approach or provide a summary.", dup_msg)),
                    tool_calls: None,
                    tool_call_id: None,
                });
                trim_messages(&mut messages);
                save_conversation().await;
                continue;
            }
            executed_calls.push(key.clone());
        }

        // Check for native tool calls (from function calling API)
        if let Some(tc) = msg_to_tool_call(&msg) {
            eprintln!(
                "{} {} {}",
                "[code-agent]".cyan(),
                "Running:".dimmed(),
                format!("{} {}", tc.name, summarize_args(&tc.arguments)).yellow()
            );

            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(response_text),
                tool_calls: msg.tool_calls.clone(),
                tool_call_id: None,
            });

            let result = tools::execute_tool(&tc.name, &tc.arguments).await;

            messages.push(ChatMessage {
                role: "tool".to_string(),
                content: Some(format!("[tool result for {}]:\n{}", tc.name, result)),
                tool_calls: None,
                tool_call_id: msg.tool_calls.as_ref().and_then(|v| v.first().map(|tc| tc.id.clone())),
            });

            trim_messages(&mut messages);
            save_conversation().await;
            continue;
        }

        // Fallback: check for text-based tool call
        let tool_call = tc_re.captures(&response_text).and_then(|caps| {
            let tag = caps.get(1)?.as_str();
            let after_tag = &response_text[caps.get(0).unwrap().end()..];
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

            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(response_text),
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

            trim_messages(&mut messages);
            save_conversation().await;
        } else {
            println!("{}", format_response(&response_text));
            save_conversation().await;
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
/// braces and strings correctly. Only returns Some if the JSON is followed
/// by a closing `</` tag.
fn extract_balanced_json(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;
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
            if depth == 0 {
                break;
            }
            depth -= 1;
            if depth == 0 {
                end = Some(start + i + 1);
                break;
            }
        }
    }
    let end = end?;
    let after = s[end..].trim_start();
    if after.starts_with("</") {
        Some(&s[start..end])
    } else {
        None
    }
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
        assert!(result.contains("-"));
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
    }

    #[test]
    fn test_format_response_empty() {
        let result = format_response("");
        assert!(result.contains("-"));
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
        assert_eq!(auto_max_tokens("hi"), 2048);
    }

    #[test]
    fn test_auto_max_tokens_long_query() {
        assert!(auto_max_tokens(&"word ".repeat(60)) >= 4096);
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
        // Genuine refusal gets penalized but not zeroed out
        assert!(score > 0.0 && score < 15.0);
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

    // -- table formatting tests --

    #[test]
    fn test_is_table_row_valid() {
        assert!(is_table_row("| a | b |"));
        assert!(is_table_row("|a|b|c|"));
        assert!(is_table_row("| **Bold** | `code` |"));
        assert!(is_table_row("|---|---|---|"));
        assert!(is_table_row("|:---|:---:|---:|"));
    }

    #[test]
    fn test_is_table_row_invalid() {
        assert!(!is_table_row("not a table"));
        assert!(!is_table_row("| just one pipe"));
        assert!(!is_table_row("this is | also not"));
        assert!(!is_table_row("|single_pipe|"));
    }

    #[test]
    fn test_is_table_separator() {
        assert!(is_table_separator("|---|---|---|"));
        assert!(is_table_separator("|:---|:---:|---:|"));
        assert!(!is_table_separator("| a | b | c |"));
    }

    #[test]
    fn test_parse_table_cells() {
        let cells = parse_table_cells("| a | bbb | cc |");
        assert_eq!(cells, vec!["a", "bbb", "cc"]);
    }

    #[test]
    fn test_parse_table_cells_no_spaces() {
        let cells = parse_table_cells("|a|b|c|");
        assert_eq!(cells, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_format_table_rows_basic() {
        let rows = vec![
            "| Name  | Age |",
            "|-------|-----|",
            "| Alice | 30  |",
            "| Bob   | 25  |",
        ];
        let result = format_table_rows(&rows, 80);
        assert!(result.contains("Alice"));
        assert!(result.contains("Bob"));
        assert!(result.contains("Name"));
        assert!(result.contains("Age"));
        // ASCII box-drawing characters
        assert!(result.contains("+"));
        assert!(result.contains("|"));
        assert!(result.contains("-"));
    }

    #[test]
    fn test_format_table_rows_no_header() {
        let rows = vec![
            "| x | y |",
            "| 1 | 2 |",
        ];
        let result = format_table_rows(&rows, 80);
        assert!(result.contains("x"));
        assert!(result.contains("y"));
        // ASCII table borders
        assert!(result.contains("+"));
        assert!(result.contains("|"));
    }

    #[test]
    fn test_format_table_in_format_response_integration() {
        let input = "Here's a comparison:\n\n| Provider | RPM |\n|----------|-----|\n| NVIDIA   | 1000+|\n| Groq     | 30   |\n";
        let result = format_response(input);
        // Table rendered with box-drawing characters
        assert!(result.contains("NVIDIA"));
        assert!(result.contains("Groq"));
        assert!(result.contains("Provider"));
        assert!(result.contains("RPM"));
        // No raw markdown pipes should remain
        assert!(!result.contains("|---"));
        // The leading "Here's a comparison:" should still be there
        assert!(result.contains("Here's a comparison"));
    }

    #[test]
    fn test_format_table_with_bold_in_cells() {
        let input = "| **Name** | Value |\n|----------|-------|\n| **CPU**  | 3.2   |\n";
        let result = format_response(input);
        assert!(result.contains("Name"));
        assert!(result.contains("CPU"));
    }

        // -- demo: before/after table rendering comparison --

    #[test]
    fn test_demo_word_wrapped_table() {
        let rows = vec![
            "| Tool        | Type                    | Key Features                                       | Language Support                  | License                      |",
            "|------------|------------------------|---------------------------------------------------|----------------------------------|------------------------------|",
            "| SonarQube   | Static Code Analysis (SAST)  | Detects security bugs, code smells, and vulnerabilities  | Java, C#, JavaScript, Python, etc.  | LGPL-3.0                     |",
            "| Semgrep     | Static Analysis (SAST)  | Lightweight, customizable rules, supports 15+ languages  | Python, Java, Go, JavaScript, etc.  | LGPL-3.0                     |",
            "| CodeQL      | Static Analysis (SAST)  | Deep semantic analysis, GitHub-native              | C/C++, Java, JavaScript, Python, etc.  | OWTFPL                       |",
            "| Bandit      | Python-specific SAST    | Focuses on Python security issues (SQLi, XSS, etc.)  | Python                            | Apache-2.0                   |",
            "| Gosec       | Go-specific SAST        | Scans Go code for security flaws                   | Go                                | Apache-2.0                   |",
            "| Trivy       | Container & Code Scanning  | Scans containers, filesystems, and repos for vulnerabilities  | Multi-language (via dependency scanning)  | Apache-2.0                   |",
            "| Snyk Code   | SAST + SCA              | Detects vulnerabilities in open-source dependencies  | JavaScript, Python, Java, etc.    | Proprietary (free tier available)  |",
        ];

        // ── OLD BEHAVIOR (simulated: no word wrapping, uniform scaling) ──
        let old_result = format_table_rows_old(&rows, 80);
        eprintln!("\nBefore (old rendering, no word wrapping):");
        for line in old_result.lines() {
            let clean = ansi_re().replace_all(line, "");
            eprintln!("{}", clean);
        }

        // ── NEW BEHAVIOR (word wrapping, proportional widths) ──
        let new_result = format_table_rows(&rows, 80);
        eprintln!("\nAfter (new rendering, with word wrapping):");
        for line in new_result.lines() {
            let clean = ansi_re().replace_all(line, "");
            eprintln!("{}", clean);
        }
        eprintln!();

        assert!(new_result.contains("Sonar"), "New: should contain wrapped content");
        assert!(new_result.contains("Detects"), "New: wrapped text present");
        assert!(new_result.contains("vulnerabilities"), "New: long text wrapped");
        assert!(old_result.contains("┌"), "Old: should have top border");
    }

    /// Old-style rendering: no word wrapping, uniform scaling, single-line cells
    fn format_table_rows_old(rows: &[&str], term_w: usize) -> String {
        if rows.is_empty() { return String::new(); }
        let mut parsed_rows: Vec<Vec<String>> = Vec::new();
        let mut has_header = false;
        for row in rows {
            let trimmed = row.trim();
            if is_table_separator(trimmed) { has_header = true; continue; }
            let cells = parse_table_cells(trimmed);
            if !cells.is_empty() { parsed_rows.push(cells); }
        }
        if parsed_rows.is_empty() { return String::new(); }
        let col_count = parsed_rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if col_count == 0 { return String::new(); }
        for row in &mut parsed_rows { while row.len() < col_count { row.push(String::new()); } }

        let mut col_widths = vec![0usize; col_count];
        for row in &parsed_rows {
            for (i, cell) in row.iter().enumerate() {
                let vl = cell.chars().count();
                if vl > col_widths[i] { col_widths[i] = vl; }
            }
        }

        // Old: uniform scaling, min 3
        let padding_total = col_count * 3 + 1;
        let total_content: usize = col_widths.iter().sum();
        let total_width = total_content + padding_total;
        if total_width > term_w && total_content > 0 {
            let available = term_w.saturating_sub(padding_total).max(col_count * 3);
            let ratio = available as f64 / total_content as f64;
            if ratio < 1.0 {
                for w in &mut col_widths { *w = (*w as f64 * ratio).max(3.0) as usize; }
            }
        }

        let a_re = ansi_re(); let ic_re = inline_code_re(); let b_re = bold_re();
        let fmt_cell = |cell: &str, w: usize, hdr: bool| -> String {
            let mut p = ic_re.replace_all(cell, |c: &regex::Captures| format!("\x1b[0;33m{}\x1b[0m", &c[1])).to_string();
            { let bc = if hdr { "93" } else { "33" };
              p = b_re.replace_all(&p, |c: &regex::Captures| format!("\x1b[1;{}m{}\x1b[0m", bc, &c[1])).to_string(); }
            let s = a_re.replace_all(&p, "").to_string();
            let pad = w.saturating_sub(s.chars().count());
            let padded = format!(" {} {:pad$} ", p, "", pad = pad);
            if hdr { format!("\x1b[1;93m{}\x1b[0m", padded) } else { format!("\x1b[0;37m{}\x1b[0m", padded) }
        };
        let hrule = |l: &str, m: &str, r: &str| -> String {
            let mut s = String::from(l);
        for (i, w) in col_widths.iter().enumerate() {
                s.push_str(&"─".repeat(*w + 2));
                if i < col_count - 1 { s.push_str(m); }
            }
            s.push_str(r); s
        };

        let mut out = String::new();
        out.push_str(&format!("\x1b[96m{}\x1b[0m\n", hrule("┌", "┬", "┐")));
        for (idx, row) in parsed_rows.iter().enumerate() {
            out.push('|');
            for (i, cell) in row.iter().enumerate() {
                out.push_str(&fmt_cell(cell, col_widths[i], has_header && idx == 0));
                out.push('|');
            }
            out.push('\n');
            if idx < parsed_rows.len() - 1 {
                out.push_str(&format!("\x1b[96m{}\x1b[0m\n", hrule("├", "┼", "┤")));
            }
        }
        out.push_str(&format!("\x1b[96m{}\x1b[0m", hrule("└", "┴", "┘")));
        out
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
