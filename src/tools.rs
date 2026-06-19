use serde_json::{json, Value};
use std::path::Path;
use std::process::Command as StdCommand;
use std::time::Duration;

pub fn all_tool_defs() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk and return its contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (creates or overwrites)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace text in an existing file (exact string replacement). If multiple matches exist, provide more surrounding context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Text to search for (must exist in file)"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Text to replace with"
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a bash command and return its output (stdout + stderr). Max timeout 30s.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents using a regex pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in (optional, defaults to current)"
                        },
                        "include": {
                            "type": "string",
                            "description": "File glob pattern to filter by (e.g. *.rs)"
                        }
                    },
                    "required": ["pattern"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match (e.g. **/*.rs)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in (optional, defaults to current)"
                        }
                    },
                    "required": ["pattern"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the directory"
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
    ]
}

pub async fn execute_tool(name: &str, args: &Value) -> String {
    match name {
        "read_file" => exec_read_file(args),
        "write_file" => exec_write_file(args),
        "edit_file" => exec_edit_file(args),
        "bash" => exec_bash(args).await,
        "grep" => exec_grep(args),
        "glob" => exec_glob(args),
        "list_dir" => exec_list_dir(args),
        _ => format!("Unknown tool: {}", name),
    }
}

fn exec_read_file(args: &Value) -> String {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: path".to_string(),
    };
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            let shown = lines.len().min(2000);
            let mut result = String::new();
            for line in &lines[..shown] {
                result.push_str(line);
                result.push('\n');
            }
            if total > shown {
                result.push_str(&format!("... ({} lines truncated)", total - shown));
            }
            result
        }
        Err(e) => format!("Error reading file: {}", e),
    }
}

fn exec_write_file(args: &Value) -> String {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: path".to_string(),
    };
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return "Missing argument: content".to_string(),
    };
    match std::fs::write(path, content) {
        Ok(_) => format!("Written {} bytes to {}", content.len(), path),
        Err(e) => format!("Error writing file: {}", e),
    }
}

fn exec_edit_file(args: &Value) -> String {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: path".to_string(),
    };
    let old = match args.get("old_string").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return "Missing argument: old_string".to_string(),
    };
    let new = match args.get("new_string").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return "Missing argument: new_string".to_string(),
    };
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Error reading file: {}", e),
    };
    if !content.contains(old) {
        return format!("Error: old_string not found in {}", path);
    }
    let new_content = content.replace(old, new);
    match std::fs::write(path, &new_content) {
        Ok(_) => format!("Replaced text in {}", path),
        Err(e) => format!("Error writing file: {}", e),
    }
}

async fn exec_bash(args: &Value) -> String {
    let cmd = match args.get("command").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return "Missing argument: command".to_string(),
    };
    let output = tokio::time::timeout(Duration::from_secs(30), async {
        StdCommand::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
    })
    .await;

    match output {
        Ok(Ok(out)) => {
            let mut result = String::new();
            if !out.stdout.is_empty() {
                result.push_str(&String::from_utf8_lossy(&out.stdout));
            }
            if !out.stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            if result.is_empty() {
                result.push_str("(no output)");
            }
            // Truncate to 10000 chars
            if result.len() > 10000 {
                result.truncate(10000);
                result.push_str("\n... (output truncated)");
            }
            result
        }
        Ok(Err(e)) => format!("Command error: {}", e),
        Err(_) => "Command timed out after 30s".to_string(),
    }
}

fn exec_grep(args: &Value) -> String {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: pattern".to_string(),
    };
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    let include = args.get("include").and_then(|v| v.as_str());

    let mut cmd = StdCommand::new("rg");
    cmd.arg("-n");
    if let Some(inc) = include {
        cmd.arg("--glob");
        cmd.arg(inc);
    }
    cmd.arg(pattern);
    cmd.arg(path);

    let max_results = 50;
    cmd.arg("-m");
    cmd.arg(max_results.to_string());

    match cmd.output() {
        Ok(out) => {
            let mut result = String::new();
            if !out.stdout.is_empty() {
                result.push_str(&String::from_utf8_lossy(&out.stdout));
            }
            if !out.stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            if result.is_empty() {
                result.push_str("(no matches)");
            }
            if result.len() > 5000 {
                result.truncate(5000);
                result.push_str("\n... (results truncated)");
            }
            result
        }
        Err(e) => format!("grep error: {} (is ripgrep installed?)", e),
    }
}

fn exec_glob(args: &Value) -> String {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: pattern".to_string(),
    };
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let mut cmd = StdCommand::new("find");
    cmd.arg(path);
    if let Some(dir) = Path::new(pattern).parent() {
        if !dir.to_string_lossy().is_empty() {
            cmd.arg("-path");
            cmd.arg(format!("*/{}", dir.to_string_lossy()));
            cmd.arg("-prune");
            cmd.arg("-o");
        }
    }
    cmd.arg("-name");
    cmd.arg(Path::new(pattern)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| pattern.to_string()));

    match cmd.output() {
        Ok(out) => {
            let result = String::from_utf8_lossy(&out.stdout).to_string();
            if result.is_empty() {
                "(no matches)".to_string()
            } else {
                let lines: Vec<&str> = result.lines().collect();
                if lines.len() > 100 {
                    let truncated: Vec<&str> = lines[..100].to_vec();
                    format!("{}\n... ({} more matches)", truncated.join("\n"), lines.len() - 100)
                } else {
                    result
                }
            }
        }
        Err(e) => format!("find error: {}", e),
    }
}

fn exec_list_dir(args: &Value) -> String {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: path".to_string(),
    };
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut items: Vec<String> = Vec::new();
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    items.push(format!("{}/", name));
                } else {
                    items.push(name);
                }
            }
            items.sort();
            if items.is_empty() {
                "(empty directory)".to_string()
            } else {
                items.join("\n")
            }
        }
        Err(e) => format!("Error reading directory: {}", e),
    }
}
