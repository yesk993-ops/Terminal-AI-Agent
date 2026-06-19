use colored::*;
use terminal_ai_agent::*;
use tokio::io::AsyncBufReadExt;

/// Parses CLI arguments. Returns (query, temperature).
fn parse_args() -> (String, f32) {
    let raw: Vec<String> = std::env::args().collect();
    let mut query = String::new();
    let mut temperature = 0.3f32;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--temperature" | "--temp" => {
                if i + 1 < raw.len() {
                    temperature = raw[i + 1].parse().unwrap_or(0.3);
                    i += 1;
                }
            }
            _ => {
                if !query.is_empty() {
                    query.push(' ');
                }
                query.push_str(&raw[i]);
            }
        }
        i += 1;
    }
    (query, temperature.clamp(0.0, 2.0))
}

#[tokio::main]
async fn main() {
    load_conversation();

    let client = reqwest::Client::new();

    // Ctrl+C handler
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        let _ = shutdown_tx.send(());
    });

    let (query, temperature) = parse_args();
    if !query.is_empty() {
        process_query(&client, &query, temperature).await;
        save_conversation();
        return;
    }

    println!(
        "{}",
        "Terminal AI Agent (type 'exit' to quit)"
            .green()
            .bold()
    );

    let stdin = tokio::io::stdin();
    let reader = tokio::io::BufReader::new(stdin);
    let mut lines = reader.lines();

    loop {
        tokio::select! {
            _ = &mut shutdown_rx => {
                println!("\n{}", "Saving conversation...".yellow());
                save_conversation();
                println!("{}", "Goodbye!".green().bold());
                break;
            }
            line = lines.next_line() => {
                match line.unwrap_or(None) {
                    None => {
                        println!();
                        break;
                    }
                    Some(ref s) if s.trim().eq_ignore_ascii_case("exit") => break,
                    Some(ref s) if s.trim().is_empty() => continue,
                    Some(s) => {
                        let trimmed = s.trim();
                        let lower = trimmed.to_ascii_lowercase();
                        if lower.starts_with("ask ") {
                            let q = trimmed[4..].trim();
                            if q.is_empty() {
                                eprintln!("{}", "Usage: ask <your question>".red());
                                continue;
                            }
                            process_query(&client, q, temperature).await;
                            save_conversation();
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

    save_conversation();
    println!("{}", "Goodbye!".green().bold());
}
