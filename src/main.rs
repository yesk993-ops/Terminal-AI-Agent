use colored::*;
use terminal_ai_agent::*;
use tokio::io::AsyncBufReadExt;

/// Parses CLI arguments. Returns (query, temperature, code_mode).
fn parse_args() -> (String, f32, bool) {
    let raw: Vec<String> = std::env::args().collect();
    let mut query = String::new();
    let mut temperature = 0.8f32;
    let mut code_mode = false;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--version" | "-V" => {
                println!("terminal_ai_agent v{}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--temperature" | "--temp" => {
                if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                    temperature = raw[i + 1].parse().unwrap_or(0.8);
                    i += 1;
                }
            }
            "--code" => code_mode = true,
            _ => {
                if !query.is_empty() {
                    query.push(' ');
                }
                query.push_str(&raw[i]);
            }
        }
        i += 1;
    }
    (query, temperature.clamp(0.0, 2.0), code_mode)
}

fn print_help() {
    println!("Terminal AI Agent v{}", env!("CARGO_PKG_VERSION"));
    println!("A fast, colorful AI agent for your terminal.");
    println!();
    println!("USAGE:");
    println!("  terminal_ai_agent [OPTIONS] [QUERY]");
    println!();
    println!("OPTIONS:");
    println!("  --help, -h           Show this help message");
    println!("  --version, -V        Show version information");
    println!("  --code               Enable coding agent mode (read/write/edit/bash/grep/glob)");
    println!("  --temp, --temperature <FLOAT>  Set temperature (0.0-2.0, default: 0.8)");
    println!();
    println!("EXAMPLES:");
    println!("  terminal_ai_agent \"What is Rust?\"");
    println!("  terminal_ai_agent --code \"create a Dockerfile\"");
    println!("  terminal_ai_agent --temp 0.3 \"Explain quantum computing\"");
    println!("  terminal_ai_agent                (starts REPL mode)");
    println!();
    println!("PROVIDERS:");
    println!("  OpenRouter, Groq, Google Gemini, NVIDIA NIM, OpenCode Gateway");
    println!("  Set API keys via environment variables:");
    println!("    OPENROUTER_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, NVIDIA_API_KEY");
    println!();
    println!("REPL COMMANDS:");
    println!("  ask <question>        Send a query to the AI");
    println!("  exit                  Quit the REPL");
}

#[tokio::main]
async fn main() {
    load_conversation().await;

    // Configure reqwest with HTTP/2, connection pooling, and DNS caching
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client");

    // Ctrl+C handler
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            eprintln!("[warn] Failed to register Ctrl+C handler: {}", e);
        }
        let _ = shutdown_tx.send(());
    });

    let (query, temperature, code_mode) = parse_args();
    if !query.is_empty() {
        if code_mode {
            process_code_query(&client, &query, temperature).await;
        } else {
            process_query(&client, &query, temperature).await;
        }
        force_save_conversation().await;
        return;
    }

    if code_mode {
        clear_conversation().await;
    }

    println!(
        "{}",
        if code_mode {
            "Terminal AI Agent - Coding Mode (type 'exit' to quit)"
        } else {
            "Terminal AI Agent (type 'exit' to quit)"
        }
        .green()
    );

    let stdin = tokio::io::stdin();
    let reader = tokio::io::BufReader::new(stdin);
    let mut lines = reader.lines();

    loop {
        tokio::select! {
            _ = &mut shutdown_rx => {
                println!("\n{}", "Saving conversation...".yellow());
                force_save_conversation().await;
                println!("{}", "Goodbye!".green());
                break;
            }
            line = lines.next_line() => {
                let line = match line {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[error] Failed to read stdin: {}", e);
                        break;
                    }
                };
                match line {
                    None => {
                        println!();
                        break;
                    }
                    Some(ref s) if s.trim().eq_ignore_ascii_case("exit") => break,
                    Some(ref s) if s.trim().is_empty() => continue,
                    Some(ref s) if s.trim().eq_ignore_ascii_case("suggest") => {
                        let suggestions = get_suggestions().await;
                        if suggestions.is_empty() {
                            eprintln!("{}", "No suggestions available yet. Ask a question first.".yellow());
                        } else {
                            println!("{}", "💡 Follow-up questions:".green());
                            for (i, sug) in suggestions.iter().enumerate() {
                                println!("  {}. {}", i + 1, sug.cyan());
                            }
                        }
                        continue;
                    }
                    Some(s) => {
                        let trimmed = s.trim();
                        let lower = trimmed.to_ascii_lowercase();
                        if lower.starts_with("ask") {
                            let q = if lower.len() > 3 && lower.as_bytes()[3] == b' ' {
                                trimmed[4..].trim()
                            } else {
                                trimmed[3..].trim()
                            };
                            if q.is_empty() {
                                eprintln!("{}", "Usage: ask <your question>".red());
                                continue;
                            }
                            if code_mode {
                                process_code_query(&client, q, temperature).await;
                            } else {
                                process_query(&client, q, temperature).await;
                            }
                            save_conversation().await;
                        } else {
                            eprintln!(
                                "{}",
                                "Unrecognized command. Use 'ask <question>' or 'exit'."
                                    .red()
                            );
                        }
                    }
                }
            }
        }
    }

    force_save_conversation().await;
    println!("{}", "Goodbye!".green());
}
