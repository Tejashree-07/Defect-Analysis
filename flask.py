# ============================================================
# FAB ENGINEER CHAT INTERFACE
# ============================================================

import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import anthropic

# ============================================================
# LOAD DATA + BUILD SUMMARY CONTEXT
# ============================================================

EXCEL_PATH = "digital_twin_results.xlsx"

def load_fab_data(path=EXCEL_PATH):
    """
    Loads the Excel results and builds a structured summary
    that gets passed to Claude as context for every question.
    """
    df = pd.read_excel(path)

    total_high_risk = len(df)
    total_wafers    = 650   

    # Top root causes
    root_counts = (
        df["root_causes"]
        .str.split("|")
        .explode()
        .str.strip()
    )
    top_causes = root_counts.value_counts().head(5)

    # Risk distribution
    high_risk_pct  = round(total_high_risk / total_wafers * 100, 1)
    normal_pct     = round(100 - high_risk_pct, 1)

    # Avg defect and join probabilities
    avg_defect = round(df["defect_prob"].mean(), 3)
    avg_join   = round(df["join_prob"].mean(), 3)
    max_defect = round(df["defect_prob"].max(), 3)

    # Most common top features
    all_features = (
        df["top_feature_1"].tolist() +
        df["top_feature_2"].tolist() +
        df["top_feature_3"].tolist()
    )
    from collections import Counter
    feature_counts = Counter(all_features).most_common(5)

    # Build context string for Claude
    context = f"""
You are an expert semiconductor process engineer AI assistant.
You are analyzing results from an agentic defect detection system
that processed {total_wafers} wafers through a Digital Twin simulator.

=== WAFER RUN SUMMARY ===
Total wafers analyzed  : {total_wafers}
High-risk wafers       : {total_high_risk} ({high_risk_pct}%)
Normal wafers          : {total_wafers - total_high_risk} ({normal_pct}%)
Average defect prob    : {avg_defect}
Average join prob      : {avg_join}
Highest defect prob    : {max_defect}

=== TOP ROOT CAUSES (across all high-risk wafers) ===
{chr(10).join([f"  {i+1}. {cause} — {count} occurrences" for i, (cause, count) in enumerate(top_causes.items())])}

=== TOP CONTRIBUTING FEATURES ===
{chr(10).join([f"  {i+1}. {feat} — {count} times" for i, (feat, count) in enumerate(feature_counts)])}

=== FULL HIGH-RISK WAFER DATA (first 30 rows) ===
{df.head(30).to_string(index=False)}

=== INSTRUCTIONS ===
Answer questions from process and equipment engineers clearly and concisely.
Use semiconductor manufacturing terminology where appropriate.
When recommending actions, be specific about which parameter and which process step.
If asked about a specific wafer or parameter, search the data above carefully.
Always relate your answers back to yield impact and process control.
"""
    return df, context

try:
    fab_df, FAB_CONTEXT = load_fab_data()
    print(f"Loaded {len(fab_df)} high-risk wafer records from {EXCEL_PATH}")
except FileNotFoundError:
    print(f"WARNING: {EXCEL_PATH} not found. Run main_code.py first.")
    fab_df      = pd.DataFrame()
    FAB_CONTEXT = "No data loaded. Please run main_code.py first to generate results."


# ============================================================
# ANTHROPIC CLIENT
# ============================================================

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable")

claude_client = anthropic.Anthropic(api_key=anthropic_api_key)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Store conversation history for multi-turn Q&A
conversation_history = []


@app.route("/")
def index():
    """Serves the chat interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Receives a question from the engineer,
    sends it to Claude with the fab data as context,
    returns the answer.
    """
    global conversation_history

    data     = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Add engineer's question to history
    conversation_history.append({
        "role": "user",
        "content": question
    })

    try:
        # Call Claude with full fab context + conversation history
        response = claude_client.messages.create (
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=FAB_CONTEXT,
            messages=conversation_history
        )
        

        answer = response.content[0].text

        # Add Claude's answer to history (multi-turn memory)
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Clears conversation history for a fresh session."""
    global conversation_history
    conversation_history = []
    return jsonify({"status": "Conversation reset"})


@app.route("/summary", methods=["GET"])
def summary():
    """Returns the auto-generated fab summary as JSON."""
    if fab_df.empty:
        return jsonify({"error": "No data loaded"})

    total_wafers   = 650
    high_risk      = len(fab_df)
    normal         = total_wafers - high_risk

    root_counts = (
        fab_df["root_causes"]
        .str.split("|")
        .explode()
        .str.strip()
        .value_counts()
        .head(3)
    )

    return jsonify({
        "total_wafers"  : total_wafers,
        "high_risk"     : high_risk,
        "normal"        : normal,
        "high_risk_pct" : round(high_risk / total_wafers * 100, 1),
        "top_causes"    : root_counts.to_dict(),
        "avg_defect_prob": round(fab_df["defect_prob"].mean(), 3),
        "max_defect_prob": round(fab_df["defect_prob"].max(), 3)
    })


# ============================================================
# HTML TEMPLATE — Professional dark theme fab dashboard
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fab Intelligence — Wafer RCA Assistant</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg:        #0a0e17;
            --surface:   #111827;
            --border:    #1e2d40;
            --accent:    #00d4ff;
            --accent2:   #ff6b35;
            --text:      #e2e8f0;
            --muted:     #64748b;
            --high-risk: #ff4444;
            --normal:    #00cc88;
            --font-mono: 'IBM Plex Mono', monospace;
            --font-sans: 'IBM Plex Sans', sans-serif;
        }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-sans);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* HEADER */
        header {
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--surface);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            background: var(--accent);
            clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
            animation: pulse 3s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .logo-text {
            font-family: var(--font-mono);
            font-size: 14px;
            color: var(--accent);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .logo-sub {
            font-size: 11px;
            color: var(--muted);
            font-family: var(--font-mono);
            letter-spacing: 1px;
        }

        .status-badge {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--normal);
            border: 1px solid var(--normal);
            padding: 4px 10px;
            border-radius: 2px;
            letter-spacing: 1px;
        }

        /* MAIN LAYOUT */
        .main {
            display: grid;
            grid-template-columns: 280px 1fr;
            flex: 1;
            overflow: hidden;
            height: calc(100vh - 65px);
        }

        /* SIDEBAR */
        .sidebar {
            border-right: 1px solid var(--border);
            padding: 24px 20px;
            overflow-y: auto;
            background: var(--surface);
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .sidebar-title {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--muted);
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        /* STAT CARDS */
        .stat-card {
            background: var(--bg);
            border: 1px solid var(--border);
            padding: 14px 16px;
            border-radius: 4px;
        }

        .stat-label {
            font-size: 10px;
            color: var(--muted);
            font-family: var(--font-mono);
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .stat-value {
            font-family: var(--font-mono);
            font-size: 22px;
            font-weight: 500;
        }

        .stat-value.high  { color: var(--high-risk); }
        .stat-value.good  { color: var(--normal); }
        .stat-value.total { color: var(--accent); }

        .stat-sub {
            font-size: 11px;
            color: var(--muted);
            margin-top: 2px;
            font-family: var(--font-mono);
        }

        /* ROOT CAUSES */
        .cause-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
        }

        .cause-item:last-child { border-bottom: none; }

        .cause-name {
            font-family: var(--font-mono);
            color: var(--text);
        }

        .cause-bar-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .cause-bar {
            height: 4px;
            background: var(--accent);
            border-radius: 2px;
            opacity: 0.7;
        }

        .cause-count {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--muted);
            min-width: 24px;
            text-align: right;
        }

        /* QUICK QUESTIONS */
        .quick-btn {
            width: 100%;
            text-align: left;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 10px 12px;
            font-size: 12px;
            font-family: var(--font-sans);
            cursor: pointer;
            border-radius: 4px;
            margin-bottom: 6px;
            transition: border-color 0.2s, color 0.2s;
            line-height: 1.4;
        }

        .quick-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* CHAT AREA */
        .chat-area {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-title {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--muted);
            letter-spacing: 1px;
        }

        .reset-btn {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--muted);
            background: none;
            border: 1px solid var(--border);
            padding: 4px 10px;
            cursor: pointer;
            border-radius: 2px;
            transition: color 0.2s, border-color 0.2s;
        }

        .reset-btn:hover {
            color: var(--accent2);
            border-color: var(--accent2);
        }

        /* MESSAGES */
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .message {
            display: flex;
            gap: 12px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .msg-avatar {
            width: 28px;
            height: 28px;
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 500;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .msg-avatar.engineer {
            background: var(--border);
            color: var(--text);
        }

        .msg-avatar.ai {
            background: var(--accent);
            color: var(--bg);
        }

        .msg-content {
            flex: 1;
        }

        .msg-role {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--muted);
            margin-bottom: 6px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .msg-text {
            font-size: 14px;
            line-height: 1.7;
            color: var(--text);
            white-space: pre-wrap;
        }

        .msg-text.ai-text {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            padding: 14px 16px;
            border-radius: 0 4px 4px 0;
        }

        /* WELCOME MESSAGE */
        .welcome {
            text-align: center;
            padding: 60px 40px;
            color: var(--muted);
        }

        .welcome h2 {
            font-family: var(--font-mono);
            font-size: 16px;
            color: var(--accent);
            margin-bottom: 12px;
            letter-spacing: 2px;
        }

        .welcome p {
            font-size: 13px;
            line-height: 1.8;
            max-width: 500px;
            margin: 0 auto;
        }

        /* INPUT AREA */
        .input-area {
            padding: 16px 24px;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
            background: var(--surface);
        }

        .input-wrap {
            flex: 1;
            position: relative;
        }

        textarea {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 12px 16px;
            font-size: 14px;
            font-family: var(--font-sans);
            border-radius: 4px;
            resize: none;
            height: 48px;
            line-height: 1.5;
            transition: border-color 0.2s;
            outline: none;
        }

        textarea:focus {
            border-color: var(--accent);
        }

        textarea::placeholder {
            color: var(--muted);
        }

        .send-btn {
            background: var(--accent);
            color: var(--bg);
            border: none;
            padding: 12px 20px;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 4px;
            letter-spacing: 1px;
            transition: opacity 0.2s;
            white-space: nowrap;
            align-self: flex-end;
        }

        .send-btn:hover   { opacity: 0.85; }
        .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        /* THINKING INDICATOR */
        .thinking {
            display: flex;
            gap: 4px;
            align-items: center;
            padding: 14px 16px;
        }

        .dot {
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            animation: bounce 1.2s infinite;
        }

        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.4; }
            40%            { transform: scale(1.2); opacity: 1; }
        }

        /* SCROLLBAR */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    </style>
</head>
<body>

<header>
    <div class="logo">
        <div class="logo-icon"></div>
        <div>
            <div class="logo-text">Fab Intelligence</div>
            <div class="logo-sub">Wafer RCA Assistant</div>
        </div>
    </div>
    <div class="status-badge">● SYSTEM ONLINE</div>
</header>

<div class="main">

    <!-- SIDEBAR -->
    <div class="sidebar">

        <div>
            <div class="sidebar-title">Run Summary</div>
            <div id="stats-container">
                <div class="stat-card" style="margin-bottom:8px">
                    <div class="stat-label">Total Wafers</div>
                    <div class="stat-value total" id="stat-total">—</div>
                </div>
                <div class="stat-card" style="margin-bottom:8px">
                    <div class="stat-label">High Risk</div>
                    <div class="stat-value high" id="stat-high">—</div>
                    <div class="stat-sub" id="stat-high-pct">— of total</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Normal</div>
                    <div class="stat-value good" id="stat-normal">—</div>
                </div>
            </div>
        </div>

        <div>
            <div class="sidebar-title">Top Root Causes</div>
            <div id="causes-container">
                <div style="font-size:12px; color:var(--muted)">Loading...</div>
            </div>
        </div>

        <div>
            <div class="sidebar-title">Quick Questions</div>
            <button class="quick-btn" onclick="askQuick('Which process parameter caused the most defects?')">
                Which parameter caused the most defects?
            </button>
            <button class="quick-btn" onclick="askQuick('Should the production line be stopped based on these results?')">
                Should we stop the production line?
            </button>
            <button class="quick-btn" onclick="askQuick('What immediate corrective actions do you recommend for the equipment engineer?')">
                What corrective actions are needed?
            </button>
            <button class="quick-btn" onclick="askQuick('Which tool type — Etching, Lithography or Deposition — had the most high-risk wafers?')">
                Which tool type is most affected?
            </button>
            <button class="quick-btn" onclick="askQuick('Summarise the overall fab health in 3 bullet points.')">
                Summarise overall fab health
            </button>
        </div>

    </div>

    <!-- CHAT -->
    <div class="chat-area">

        <div class="chat-header">
            <div class="chat-title">ENGINEER Q&A — Ask anything about the wafer run</div>
            <button class="reset-btn" onclick="resetChat()">CLEAR SESSION</button>
        </div>

        <div class="messages" id="messages">
            <div class="welcome">
                <h2>WAFER RCA ASSISTANT</h2>
                <p>
                    Ask me anything about this wafer run in plain English.<br>
                    I have full access to all 650 wafer results, defect probabilities,
                    root causes, and recommended actions.
                </p>
            </div>
        </div>

        <div class="input-area">
            <div class="input-wrap">
                <textarea
                    id="question-input"
                    placeholder="Ask a question — e.g. 'Which parameter drifted the most?' or 'What should I check first?'"
                    onkeydown="handleKey(event)"
                ></textarea>
            </div>
            <button class="send-btn" id="send-btn" onclick="sendQuestion()">SEND →</button>
        </div>

    </div>
</div>

<script>
    // Load summary on page load
    async function loadSummary() {
        try {
            const res  = await fetch("/summary");
            const data = await res.json();

            document.getElementById("stat-total").textContent  = data.total_wafers.toLocaleString();
            document.getElementById("stat-high").textContent   = data.high_risk.toLocaleString();
            document.getElementById("stat-high-pct").textContent = data.high_risk_pct + "% of total";
            document.getElementById("stat-normal").textContent = data.normal.toLocaleString();

            // Root causes
            const causesEl = document.getElementById("causes-container");
            causesEl.innerHTML = "";

            const causes = data.top_causes;
            const maxCount = Math.max(...Object.values(causes));

            Object.entries(causes).forEach(([cause, count]) => {
                const barWidth = Math.round((count / maxCount) * 80);
                causesEl.innerHTML += `
                    <div class="cause-item">
                        <span class="cause-name">${cause.replace(/_/g," ")}</span>
                        <div class="cause-bar-wrap">
                            <div class="cause-bar" style="width:${barWidth}px"></div>
                            <span class="cause-count">${count}</span>
                        </div>
                    </div>`;
            });

        } catch(e) {
            console.error("Failed to load summary:", e);
        }
    }

    // Send question to backend
    async function sendQuestion() {
        const input   = document.getElementById("question-input");
        const question = input.value.trim();
        if (!question) return;

        const messagesEl = document.getElementById("messages");
        const sendBtn    = document.getElementById("send-btn");

        // Remove welcome message if present
        const welcome = messagesEl.querySelector(".welcome");
        if (welcome) welcome.remove();

        // Show engineer message
        messagesEl.innerHTML += `
            <div class="message">
                <div class="msg-avatar engineer">ENG</div>
                <div class="msg-content">
                    <div class="msg-role">Process Engineer</div>
                    <div class="msg-text">${escapeHtml(question)}</div>
                </div>
            </div>`;

        // Show thinking indicator
        const thinkingId = "thinking-" + Date.now();
        messagesEl.innerHTML += `
            <div class="message" id="${thinkingId}">
                <div class="msg-avatar ai">AI</div>
                <div class="msg-content">
                    <div class="msg-role">Fab Intelligence</div>
                    <div class="msg-text ai-text">
                        <div class="thinking">
                            <div class="dot"></div>
                            <div class="dot"></div>
                            <div class="dot"></div>
                        </div>
                    </div>
                </div>
            </div>`;

        input.value = "";
        sendBtn.disabled = true;
        messagesEl.scrollTop = messagesEl.scrollHeight;

        try {
            const res  = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });

            const data = await res.json();

            // Replace thinking with answer
            const thinkingEl = document.getElementById(thinkingId);
            if (data.answer) {
                thinkingEl.querySelector(".msg-text").textContent = data.answer;
                thinkingEl.querySelector(".msg-text").classList.add("ai-text");
            } else {
                thinkingEl.querySelector(".msg-text").textContent = "Error: " + (data.error || "Unknown error");
            }

        } catch(e) {
            document.getElementById(thinkingId).querySelector(".msg-text").textContent = "Connection error. Is the server running?";
        }

        sendBtn.disabled = false;
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function askQuick(question) {
        document.getElementById("question-input").value = question;
        sendQuestion();
    }

    function handleKey(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    }

    async function resetChat() {
        await fetch("/reset", { method: "POST" });
        const messagesEl = document.getElementById("messages");
        messagesEl.innerHTML = `
            <div class="welcome">
                <h2>WAFER RCA ASSISTANT</h2>
                <p>Session cleared. Ask me anything about the wafer run.</p>
            </div>`;
    }

    function escapeHtml(text) {
        return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    // Init
    loadSummary();
</script>

</body>
</html>
"""

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  FAB INTELLIGENCE — Chat Interface")
    print("="*50)
    print("  Open your browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
