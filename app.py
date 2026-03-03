import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile

# Use a clean plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# --- SECTION 1: CONSTANTS LOOKUP ---
SPC_CONSTANTS_DEFAULTS = {
    2: (1.880, 0.0, 3.267), 3: (1.023, 0.0, 2.574), 4: (0.729, 0.0, 2.282),
    5: (0.577, 0.0, 2.114), 6: (0.483, 0.0, 2.004), 7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864), 9: (0.337, 0.184, 1.816), 10: (0.308, 0.223, 1.777)
}

# --- SECTION 2: HELPER FUNCTIONS ---
def get_defaults(n_val):
    n = int(n_val)
    return SPC_CONSTANTS_DEFAULTS.get(n, (0.577, 0.0, 2.114))

def parse_data(text):
    try:
        if not text: return None
        clean = text.replace(",", "\n").replace(" ", "\n").replace("\t", "\n")
        return np.array([float(x) for x in clean.split() if x.strip()])
    except:
        return None

def check_control(data, ucl, lcl, chart_name):
    violations = []
    if isinstance(ucl, (int, float)):
        violations = [i+1 for i, x in enumerate(data) if x > ucl or x < lcl]
    else:
        violations = [i+1 for i, x in enumerate(data) if x > ucl[i] or x < lcl[i]]
    
    if violations:
        return f"❌ {chart_name}: OUT OF CONTROL at points {violations}", violations
    else:
        return f"✅ {chart_name}: Process is Stable.", []

def create_plot(data, ucl, lcl, cl, title, violations, ylabel="Value"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data, 'o-', color='#1f77b4', label='Data')
    
    if isinstance(ucl, (int, float)):
        ax.axhline(ucl, color='red', linestyle='--', label=f'UCL={ucl:.3f}')
        ax.axhline(lcl, color='red', linestyle='--', label=f'LCL={lcl:.3f}')
        ax.axhline(cl, color='green', label=f'CL={cl:.3f}')
        ax.fill_between(range(len(data)), lcl, ucl, color='green', alpha=0.05)
    else:
        ax.step(range(len(data)), ucl, color='red', linestyle='--', where='mid', label='Variable UCL')
        ax.step(range(len(data)), lcl, color='red', linestyle='--', where='mid', label='Variable LCL')
        ax.axhline(cl, color='green', label=f'CL={cl:.3f}')

    if violations:
        idx = [v-1 for v in violations]
        ax.scatter(idx, data[idx], color='red', s=150, zorder=10, edgecolor='white', label='Violation')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Sample Number")
    ax.legend(loc='upper right')
    return fig

# --- SECTION 3: CALCULATION LOGIC ---
def calculate_charts(chart_type, data_input, n_val, a2, d3, d4, size_input):
    try:
        data = parse_data(data_input)
        if data is None or len(data) == 0:
            return None, "Error: No valid data entered.", "", None

        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('#fafafa')
        final_msg = ""
        stats_text = ""
        n = int(n_val)

        # === 1. MEAN & RANGE CHART (X-bar & R) ===
        if chart_type == "Mean & Range (X-bar & R)":
            if len(data) < n:
                return None, f"Error: Data length ({len(data)}) is smaller than sample size ({n}).", "", None
            
            num_groups = len(data) // n
            clean_data = data[:num_groups * n]
            groups = clean_data.reshape((num_groups, n))
            
            x_bars = np.mean(groups, axis=1)
            ranges = np.ptp(groups, axis=1)
            x_grand = np.mean(x_bars)
            r_bar = np.mean(ranges)
            
            # Limits using USER CONSTANTS
            ucl_x = x_grand + (a2 * r_bar)
            lcl_x = x_grand - (a2 * r_bar)
            ucl_r = d4 * r_bar
            lcl_r = d3 * r_bar
            
            msg_x, viol_x = check_control(x_bars, ucl_x, lcl_x, "X-bar")
            msg_r, viol_r = check_control(ranges, ucl_r, lcl_r, "Range")
            final_msg = f"{msg_x}\n{msg_r}"
            stats_text = (f"X-bar: UCL={ucl_x:.3f}, CL={x_grand:.3f}, LCL={lcl_x:.3f}\n"
                          f"Range: UCL={ucl_r:.3f}, CL={r_bar:.3f}, LCL={lcl_r:.3f}")
            
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(x_bars, 'o-', color='blue', label='Means')
            ax1.axhline(ucl_x, c='red', ls='--'); ax1.axhline(lcl_x, c='red', ls='--'); ax1.axhline(x_grand, c='green')
            if viol_x: ax1.scatter([v-1 for v in viol_x], x_bars[[v-1 for v in viol_x]], c='red', s=100)
            ax1.set_title("X-bar Chart")
            
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(ranges, 'o-', color='orange', label='Ranges')
            ax2.axhline(ucl_r, c='red', ls='--'); ax2.axhline(lcl_r, c='red', ls='--'); ax2.axhline(r_bar, c='green')
            if viol_r: ax2.scatter([v-1 for v in viol_r], ranges[[v-1 for v in viol_r]], c='red', s=100)
            ax2.set_title("R Chart")
            plt.tight_layout()

        # === 2. P CHART ===
        elif chart_type == "P Chart (Fraction Defective)":
            if np.any(data > 1): p_vals = data / n
            else: p_vals = data
            
            p_bar = np.mean(p_vals)
            sigma = np.sqrt(p_bar * (1 - p_bar) / n)
            ucl = min(1.0, p_bar + 3 * sigma)
            lcl = max(0.0, p_bar - 3 * sigma)
            
            msg, viol = check_control(p_vals, ucl, lcl, "P Chart")
            final_msg = msg
            stats_text = f"P-bar: {p_bar:.4f} | UCL: {ucl:.4f} | LCL: {lcl:.4f}"
            fig = create_plot(p_vals, ucl, lcl, p_bar, f"P Chart (n={n})", viol, "Fraction Defective")

        # === 3. NP CHART ===
        elif chart_type == "np Chart (Number of Defectives)":
            np_bar = np.mean(data)
            p_bar = np_bar / n
            sigma = np.sqrt(n * p_bar * (1 - p_bar))
            ucl = np_bar + 3 * sigma
            lcl = max(0.0, np_bar - 3 * sigma)
            
            msg, viol = check_control(data, ucl, lcl, "np Chart")
            final_msg = msg
            stats_text = f"np-bar: {np_bar:.2f} | UCL: {ucl:.2f} | LCL: {lcl:.2f}"
            fig = create_plot(data, ucl, lcl, np_bar, f"np Chart (n={n})", viol, "Count of Defectives")

        # === 4. C CHART ===
        elif chart_type == "C Chart (Defects per Unit)":
            c_bar = np.mean(data)
            ucl = c_bar + 3 * np.sqrt(c_bar)
            lcl = max(0.0, c_bar - 3 * np.sqrt(c_bar))
            
            msg, viol = check_control(data, ucl, lcl, "C Chart")
            final_msg = msg
            stats_text = f"C-bar: {c_bar:.2f} | UCL: {ucl:.2f} | LCL: {lcl:.2f}"
            fig = create_plot(data, ucl, lcl, c_bar, "C Chart", viol, "Defect Count")

        # === 5. U CHART ===
        elif chart_type == "U Chart (Variable Sample Size)":
            sizes = parse_data(size_input)
            if sizes is None or len(sizes) != len(data):
                return None, "Error: 'Sample Sizes' list length must match 'Data' length.", "", None
            u_vals = data / sizes
            u_bar = np.sum(data) / np.sum(sizes)
            sigmas = np.sqrt(u_bar / sizes)
            ucl = u_bar + 3 * sigmas
            lcl = np.maximum(0, u_bar - 3 * sigmas)
            msg, viol = check_control(u_vals, ucl, lcl, "U Chart")
            final_msg = msg
            stats_text = f"U-bar (Weighted): {u_bar:.4f}\n(Limits vary per sample)"
            fig = create_plot(u_vals, ucl, lcl, u_bar, "U Chart (Variable n)", viol, "Defects per Unit")

        # PDF GENERATION (EMOJI SAFE)
        # 1. Save Chart Image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name)
        
        # 2. Clean Text for PDF (Remove Emojis to fix Latin-1 Error)
        clean_msg = final_msg.replace("✅", "[PASS]").replace("❌", "[FAIL]")
        clean_stats = stats_text.replace("✅", "").replace("❌", "")

        # 3. Build PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"SQC Report: {chart_type}", ln=True, align='C')
        
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"\nInterpretation:\n{clean_msg}\n\nStatistics:\n{clean_stats}")
        
        pdf.image(tmp.name, x=10, y=70, w=190)
        pdf_path = tempfile.mktemp(suffix=".pdf")
        pdf.output(pdf_path)

        return fig, final_msg, stats_text, pdf_path

    except Exception as e:
        return None, f"System Error: {str(e)}", "", None

# --- SECTION 4: UI ---
theme = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(theme=theme, title="SQC Pro") as app:
    gr.Markdown("# 🏭 Advanced Statistical Quality Control")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Configuration")
            chart_sel = gr.Dropdown(
                ["Mean & Range (X-bar & R)", "P Chart (Fraction Defective)", 
                 "np Chart (Number of Defectives)", "C Chart (Defects per Unit)", 
                 "U Chart (Variable Sample Size)"], 
                label="Select Chart Type", value="Mean & Range (X-bar & R)"
            )
            
            # Constants Row
            with gr.Group(visible=True) as const_group:
                gr.Markdown("### Control Constants (Auto-filled)")
                with gr.Row():
                    a2_box = gr.Number(label="A2", value=0.729)
                    d3_box = gr.Number(label="D3", value=0.0)
                    d4_box = gr.Number(label="D4", value=2.282)

            gr.Markdown("### 2. Data Entry")
            data_box = gr.Textbox(
                label="Data (Observations / Counts)", 
                placeholder="Paste numbers here...",
                lines=5,
                value="12, 14, 13, 15, 11, 12, 14, 13, 15, 14, 16, 15, 13, 12, 14, 13, 14, 15, 13, 16"
            )
            
            size_box = gr.Textbox(label="Variable Sample Sizes (for U Chart only)", placeholder="200, 150, 300...", visible=False)
            n_box = gr.Number(label="Sample Size (n)", value=4)
            
            btn = gr.Button("🚀 Generate Chart", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 3. Analysis Results")
            interp_box = gr.Textbox(label="Interpretation", elem_classes="output-box")
            plot_res = gr.Plot(label="Chart Visualization")
            stats_res = gr.Textbox(label="Detailed Statistics", lines=2)
            pdf_res = gr.File(label="Download Report")

    def update_ui(chart_type, n_val):
        show_const = False
        show_sizes = False
        show_n = True
        
        if chart_type == "Mean & Range (X-bar & R)":
            show_const = True
        elif chart_type == "U Chart (Variable Sample Size)":
            show_sizes = True
            show_n = False
        elif chart_type == "C Chart (Defects per Unit)":
            show_n = False
            
        a2, d3, d4 = get_defaults(n_val)
        
        return {
            const_group: gr.Group(visible=show_const),
            size_box: gr.Textbox(visible=show_sizes),
            n_box: gr.Number(visible=show_n),
            a2_box: gr.Number(value=a2),
            d3_box: gr.Number(value=d3),
            d4_box: gr.Number(value=d4)
        }

    chart_sel.change(update_ui, [chart_sel, n_box], [const_group, size_box, n_box, a2_box, d3_box, d4_box])
    n_box.change(update_ui, [chart_sel, n_box], [const_group, size_box, n_box, a2_box, d3_box, d4_box])
    
    btn.click(calculate_charts, 
              inputs=[chart_sel, data_box, n_box, a2_box, d3_box, d4_box, size_box],
              outputs=[plot_res, interp_box, stats_res, pdf_res])

app.launch()
      
