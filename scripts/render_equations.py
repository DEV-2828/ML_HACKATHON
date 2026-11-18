"""
Render mathematical equations to high-DPI transparent PNGs for ReportLab inclusion.
Uses Matplotlib mathtext parser.
"""

import matplotlib.pyplot as plt
from pathlib import Path

def render_eq(latex_expr: str, out_file: Path, fontsize: int = 15):
    plt.close('all')
    fig = plt.figure(figsize=(7, 0.9), dpi=300)
    fig.patch.set_alpha(0.0)
    plt.axis('off')
    plt.text(0.5, 0.5, f"${latex_expr}$", fontsize=fontsize, ha='center', va='center', color='#1E293B')
    plt.savefig(str(out_file), bbox_inches='tight', pad_inches=0.08, transparent=True)
    plt.close(fig)

if __name__ == '__main__':
    plot_dir = Path("PLOT")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    render_eq(
        r"P(z_t \mid r_i) = \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left(-\frac{d(z_t, r_i)^2}{2\sigma_z^2}\right)",
        plot_dir / "eq_emission.png"
    )
    
    render_eq(
        r"P(r_j \mid r_i) = \frac{1}{\beta} \exp\left(-\frac{|d_{\mathrm{road}}(r_i, r_j) - d_{\mathrm{euclidean}}(z_t, z_{t+1})|}{\beta}\right)",
        plot_dir / "eq_transition.png"
    )
    
    render_eq(
        r"V(t, j) = \max_{i} \left[ V(t-1, i) + \ln P(r_j \mid r_i) \right] + \ln P(z_t \mid r_j)",
        plot_dir / "eq_viterbi.png"
    )
    
    print("Successfully rendered all 3 equations with distinct content!")
