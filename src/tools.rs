use serde_json::{json, Value};
use std::process::Command as StdCommand;
use std::time::Duration;

use glob::glob as glob_match;
use tokio::time::timeout;

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
        "grep" => exec_grep(args).await,
        "glob" => exec_glob(args).await,
        "list_dir" => exec_list_dir(args),
        _ => format!("Unknown tool: {}", name),
    }
}

fn exec_read_file(args: &Value) -> String {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: path".to_string(),
    };
    
    // Use BufReader for streaming large files instead of loading entire file into memory
    match std::fs::File::open(path) {
        Ok(file) => {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(file);
            let mut result = String::new();
            let mut line_count = 0;
            const MAX_LINES: usize = 2000;
            
            for line in reader.lines().take(MAX_LINES) {
                match line {
                    Ok(l) => {
                        result.push_str(&l);
                        result.push('\n');
                        line_count += 1;
                    }
                    Err(e) => return format!("Error reading file: {}", e),
                }
            }
            
            // Check if there are more lines without loading them all
            if line_count == MAX_LINES {
                result.push_str(&format!("... (more lines truncated)"));
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
    let count = content.matches(old).count();
    let new_content = content.replacen(old, new, 1);
    match std::fs::write(path, &new_content) {
        Ok(_) => {
            if count > 1 {
                format!(
                    "Replaced 1 of {} occurrences in {} (provide more context to target a specific one)",
                    count, path
                )
            } else {
                format!("Replaced text in {}", path)
            }
        }
        Err(e) => format!("Error writing file: {}", e),
    }
}

async fn exec_bash(args: &Value) -> String {
    let cmd = match args.get("command").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return "Missing argument: command".to_string(),
    };
    let cmd = cmd.to_string();

    let output = timeout(
        Duration::from_secs(30),
        tokio::task::spawn_blocking(move || {
            StdCommand::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()
        }),
    )
    .await;

    match output {
        Ok(Ok(Ok(out))) => {
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
        Ok(Ok(Err(e))) => format!("Command error: {}", e),
        Ok(Err(_)) => "Failed to spawn command".to_string(),
        Err(_) => "Command timed out after 30s".to_string(),
    }
}

async fn exec_grep(args: &Value) -> String {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: pattern".to_string(),
    };
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".")
        .to_string();
    let include = args.get("include").and_then(|v| v.as_str()).map(|s| s.to_string());

    let pattern = pattern.to_string();
    let result = tokio::task::spawn_blocking(move || {
        // Try ripgrep first (fastest), fall back to grep -rn
        let has_rg = StdCommand::new("rg")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .ok()
            .map(|s| s.success())
            .unwrap_or(false);

        if has_rg {
            let mut cmd = StdCommand::new("rg");
            cmd.arg("-n");
            if let Some(ref inc) = include {
                cmd.arg("--glob");
                cmd.arg(inc);
            }
            cmd.arg(&pattern);
            cmd.arg(&path);
            cmd.arg("-m").arg("50");
            return cmd.output();
        }

        // Fallback: use grep -rn with --include if needed
        let mut cmd = StdCommand::new("grep");
        cmd.arg("-rn");
        if let Some(ref inc) = include {
            cmd.arg("--include");
            cmd.arg(inc);
        }
        cmd.arg(&pattern);
        cmd.arg(&path);
        cmd.arg("-m").arg("50");
        cmd.output()
    })
    .await;

    match result {
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
                result.push_str("(no matches)");
            }
            if result.len() > 5000 {
                result.truncate(5000);
                result.push_str("\n... (results truncated)");
            }
            result
        }
        Ok(Err(e)) => format!("grep error: {}", e),
        Err(_) => "Failed to spawn grep command".to_string(),
    }
}

async fn exec_glob(args: &Value) -> String {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Missing argument: pattern".to_string(),
    };
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".")
        .to_string();

    let full_pattern = if path == "." {
        pattern.to_string()
    } else {
        format!("{}/{}", path.trim_end_matches('/'), pattern.trim_start_matches('/'))
    };

    match tokio::task::spawn_blocking(move || {
        let mut results: Vec<String> = Vec::new();
        match glob_match(&full_pattern) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    results.push(entry.to_string_lossy().to_string());
                }
                results.sort();
                if results.is_empty() {
                    "(no matches)".to_string()
                } else if results.len() > 100 {
                    let truncated: Vec<String> = results[..100].to_vec();
                    format!(
                        "{}\n... ({} more matches)",
                        truncated.join("\n"),
                        results.len() - 100
                    )
                } else {
                    results.join("\n")
                }
            }
            Err(e) => format!("Glob error: {}", e),
        }
    })
    .await
    {
        Ok(result) => result,
        Err(_) => "Failed to spawn glob task".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::io::Write;

    // --- read_file ---

    #[test]
    fn test_read_file_success() {
        let dir = std::env::temp_dir().join("tai_test_read");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test.txt");
        fs::write(&file_path, "hello\nworld\n").unwrap();

        let args = json!({"path": file_path.to_str().unwrap()});
        let result = exec_read_file(&args);
        assert!(result.contains("hello"));
        assert!(result.contains("world"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_file_missing() {
        let args = json!({"path": "/nonexistent/path/file.txt"});
        let result = exec_read_file(&args);
        assert!(result.starts_with("Error reading file:"));
    }

    #[test]
    fn test_read_file_missing_arg() {
        let args = json!({});
        assert_eq!(exec_read_file(&args), "Missing argument: path");
    }

    #[test]
    fn test_read_file_truncation() {
        let dir = std::env::temp_dir().join("tai_test_trunc");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("big.txt");
        let mut f = fs::File::create(&file_path).unwrap();
        for i in 0..2100 {
            writeln!(f, "line {}", i).unwrap();
        }
        drop(f);

        let args = json!({"path": file_path.to_str().unwrap()});
        let result = exec_read_file(&args);
        assert!(result.contains("truncated"));

        fs::remove_dir_all(&dir).ok();
    }

    // --- write_file ---

    #[test]
    fn test_write_file_success() {
        let dir = std::env::temp_dir().join("tai_test_write");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("out.txt");
        let content = "hello from test";

        let args = json!({"path": file_path.to_str().unwrap(), "content": content});
        let result = exec_write_file(&args);
        assert!(result.contains("Written"));
        assert!(result.contains(&content.len().to_string()));

        let read_back = fs::read_to_string(&file_path).unwrap();
        assert_eq!(read_back, content);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_file_missing_path() {
        let args = json!({"content": "foo"});
        assert_eq!(exec_write_file(&args), "Missing argument: path");
    }

    #[test]
    fn test_write_file_missing_content() {
        let args = json!({"path": "/tmp/foo"});
        assert_eq!(exec_write_file(&args), "Missing argument: content");
    }

    // --- edit_file ---

    #[test]
    fn test_edit_file_single_occurrence() {
        let dir = std::env::temp_dir().join("tai_test_edit");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("edit.txt");
        fs::write(&file_path, "before middle after").unwrap();

        let args = json!({"path": file_path.to_str().unwrap(), "old_string": "middle", "new_string": "center"});
        let result = exec_edit_file(&args);
        assert!(result.contains("Replaced text in"));

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "before center after");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_edit_file_duplicate_warning() {
        let dir = std::env::temp_dir().join("tai_test_edit_dup");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("dup.txt");
        fs::write(&file_path, "foo bar foo").unwrap();

        let args = json!({"path": file_path.to_str().unwrap(), "old_string": "foo", "new_string": "baz"});
        let result = exec_edit_file(&args);
        assert!(result.contains("1 of 2 occurrences"));

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "baz bar foo");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_edit_file_not_found() {
        let dir = std::env::temp_dir().join("tai_test_edit_nf");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("nf.txt");
        fs::write(&file_path, "hello").unwrap();

        let args = json!({"path": file_path.to_str().unwrap(), "old_string": "nope", "new_string": "yes"});
        let result = exec_edit_file(&args);
        assert!(result.contains("not found"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_edit_file_missing_args() {
        assert_eq!(exec_edit_file(&json!({})), "Missing argument: path");
        assert_eq!(exec_edit_file(&json!({"path": "/x"})), "Missing argument: old_string");
        assert_eq!(exec_edit_file(&json!({"path": "/x", "old_string": "a"})), "Missing argument: new_string");
    }

    // --- bash ---

    #[tokio::test]
    async fn test_bash_echo() {
        let args = json!({"command": "echo hello"});
        let result = exec_bash(&args).await;
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn test_bash_missing_arg() {
        let args = json!({});
        assert_eq!(exec_bash(&args).await, "Missing argument: command");
    }

    #[tokio::test]
    async fn test_bash_no_output() {
        let args = json!({"command": "true"});
        let result = exec_bash(&args).await;
        assert_eq!(result, "(no output)");
    }

    // --- grep ---

    #[tokio::test]
    async fn test_grep_missing_arg() {
        let args = json!({});
        assert_eq!(exec_grep(&args).await, "Missing argument: pattern");
    }

    // --- glob ---

    #[tokio::test]
    async fn test_glob_finds_files() {
        let dir = std::env::temp_dir().join("tai_test_glob");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.rs"), "").unwrap();
        fs::write(dir.join("b.rs"), "").unwrap();
        fs::write(dir.join("c.txt"), "").unwrap();

        let args = json!({"pattern": "*.rs", "path": dir.to_str().unwrap()});
        let result = exec_glob(&args).await;
        assert!(result.contains("a.rs"));
        assert!(result.contains("b.rs"));
        assert!(!result.contains("c.txt"));

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let dir = std::env::temp_dir().join("tai_test_glob_empty");
        fs::create_dir_all(&dir).unwrap();

        let args = json!({"pattern": "*.xyz", "path": dir.to_str().unwrap()});
        let result = exec_glob(&args).await;
        assert_eq!(result, "(no matches)");

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_glob_missing_arg() {
        let args = json!({});
        assert_eq!(exec_glob(&args).await, "Missing argument: pattern");
    }

    // --- list_dir ---

    #[test]
    fn test_list_dir_contents() {
        let dir = std::env::temp_dir().join("tai_test_list");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("file1.txt"), "").unwrap();
        fs::create_dir(dir.join("subdir")).unwrap();

        let args = json!({"path": dir.to_str().unwrap()});
        let result = exec_list_dir(&args);
        assert!(result.contains("file1.txt"));
        assert!(result.contains("subdir/"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_dir_empty() {
        let dir = std::env::temp_dir().join("tai_test_list_empty");
        fs::create_dir_all(&dir).unwrap();

        let args = json!({"path": dir.to_str().unwrap()});
        let result = exec_list_dir(&args);
        assert_eq!(result, "(empty directory)");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_dir_missing_arg() {
        let args = json!({});
        assert_eq!(exec_list_dir(&args), "Missing argument: path");
    }

    // --- execute_tool ---

    #[tokio::test]
    async fn test_execute_tool_unknown() {
        let result = execute_tool("nonexistent", &json!({})).await;
        assert!(result.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_execute_tool_dispatches_to_read() {
        let dir = std::env::temp_dir().join("tai_test_dispatch");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("dispatch.txt");
        fs::write(&file_path, "dispatch content").unwrap();

        let result = execute_tool("read_file", &json!({"path": file_path.to_str().unwrap()})).await;
        assert!(result.contains("dispatch content"));

        fs::remove_dir_all(&dir).ok();
    }
}
