"""Generate three separate model-architecture diagrams for train_second_order_fno.py."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────────
BG     = '#0f1117'
PANEL  = '#1c1f2e'
BLUE   = '#4a9eff'
GREEN  = '#4ecb71'
ORANGE = '#f0a050'
PURPLE = '#b07fff'
RED    = '#ff6b6b'
TEAL   = '#4dd9c0'
GREY   = '#8899aa'
WHITE  = '#e8eaf0'
YELLOW = '#f5d76e'

OUT_DIR = '/home/hl1138/TFNO/src_second_order'


# ── shared helpers ─────────────────────────────────────────────────────────────

def _make_ax(figsize):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    ax  = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, color, label, sublabel='', fontsize=9, alpha=0.92, radius=0.015):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=color, edgecolor=WHITE, linewidth=0.8,
                          alpha=alpha, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + h*0.14, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=WHITE, zorder=4)
        ax.text(x + w/2, cy - h*0.20, sublabel,
                ha='center', va='center', fontsize=fontsize - 1.5,
                color=WHITE, alpha=0.82, zorder=4)
    else:
        ax.text(x + w/2, cy, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=WHITE, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=WHITE, lw=1.4, style='->'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=5)


def txt(ax, x, y, s, fontsize=9, color=WHITE, ha='center', va='center', **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, color=color, zorder=6, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Second-order differencing concept
# ══════════════════════════════════════════════════════════════════════════════

def fig1():
    fig, ax = _make_ax((14, 5))
    fig.text(0.5, 0.97, 'Part 1 — Second-Order Differencing in Depth Space (Δ²D)',
             ha='center', va='top', fontsize=14, color=WHITE, fontweight='bold')

    t  = np.linspace(0, 10, 300)
    D  = 2.5 + 1.8*np.sin(0.5*t) + 0.3*t + 0.4*np.sin(1.5*t)
    D  = D / D.max()
    dD  = np.diff(D,  prepend=D[0])
    d2D = np.diff(dD, prepend=dD[0])
    tn  = np.linspace(0, 1, len(t))

    def plot_curve(tx, ty, tw, th, ys, color, title):
        y0, y1 = ys.min(), ys.max()
        yr = max(y1 - y0, 1e-3)
        yn = (ys - y0) / yr * th * 0.72 + ty + th * 0.14
        xn = tn * tw + tx
        ax.plot(xn, yn, color=color, lw=2.2, zorder=4)
        for yv in [ty, ty + th]:
            ax.plot([tx, tx+tw], [yv, yv], color=GREY, lw=0.5, zorder=3)
        ax.plot([tx, tx], [ty, ty+th], color=GREY, lw=0.5, zorder=3)
        txt(ax, tx+tw/2, ty+th+0.07, title, fontsize=11, color=color, fontweight='bold')
        if y0 < 0 < y1:
            yz = (0 - y0) / yr * th * 0.72 + ty + th * 0.14
            ax.plot([tx, tx+tw], [yz, yz], '--', color=WHITE, lw=0.9, alpha=0.5, zorder=3)
            txt(ax, tx - 0.015, yz, '0', fontsize=8, color=GREY)

    plot_curve(0.03, 0.12, 0.28, 0.68, D,   BLUE,   'Depth  D_t')
    plot_curve(0.37, 0.12, 0.26, 0.68, dD,  ORANGE, 'First diff  ΔD_t = D_t − D_{t−1}')
    plot_curve(0.69, 0.12, 0.28, 0.68, d2D, GREEN,  'Second diff  Δ²D_t = D_t − 2·D_{t−1} + D_{t−2}')

    arrow(ax, 0.315, 0.47, 0.365, 0.47, color=WHITE, lw=1.6)
    arrow(ax, 0.645, 0.47, 0.685, 0.47, color=WHITE, lw=1.6)

    txt(ax, 0.175, 0.22,
        'Slow upward drift\nhard for the network to learn',
        fontsize=9, color=WHITE, ha='center',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#1e2a3a', edgecolor=BLUE, alpha=0.88))

    txt(ax, 0.83, 0.22,
        'Zero-centred during dry periods\nnear-zero when no flooding event\n→ much easier to learn',
        fontsize=9, color=WHITE, ha='center',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#1e3020', edgecolor=GREEN, alpha=0.88))

    fig.savefig(f'{OUT_DIR}/diagram_1_differencing.png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved → {OUT_DIR}/diagram_1_differencing.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Input assembly + TFNO + output + reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def fig2():
    fig, ax = _make_ax((16, 6))
    fig.text(0.5, 0.97, 'Part 2 — Input Construction  →  TFNO  →  Depth Reconstruction',
             ha='center', va='top', fontsize=14, color=WHITE, fontweight='bold')

    # Layout budget (canvas 0–1):
    #  channels(0.40) | arr | TFNO(0.15) | arr | output(0.08) | arr | recon(0.19)
    #  Each arrow zone = 0.04 (0.01 gap + 0.02 arrow + 0.01 gap)
    bx0  = 0.02          # left edge of channel stack
    by0  = 0.38          # bottom y of main boxes
    bh   = 0.22          # main box height
    ch_w = 0.40          # total width of 9-channel stack
    bw   = (ch_w - 8*0.005) / 9   # ≈ 0.040 per channel box
    gap  = 0.005

    # ── 9 channel boxes ───────────────────────────────────────────────────────
    ch_colors = [TEAL]*4 + [PURPLE]*4 + [YELLOW]
    ch_labels = ['DEM', 'Manning', 'Pervious', 'Slope',
                 'Δ²D\nt-3', 'Δ²D\nt-2', 'Δ²D\nt-1', 'Δ²D\nt', 'R_t']
    for i, (col, lbl) in enumerate(zip(ch_colors, ch_labels)):
        bx = bx0 + i*(bw + gap)
        box(ax, bx, by0, bw, bh, col, lbl, fontsize=8, radius=0.01)

    # group braces
    xs_r = bx0 + 4*(bw+gap) - gap
    xd_l = xs_r + gap;  xd_r = xd_l + 4*(bw+gap) - gap
    xr_l = xd_r + gap;  xr_r = xr_l + bw

    def brace(x0, x1, yb, lbl, color):
        ax.annotate('', xy=(x0, yb-0.04), xytext=(x1, yb-0.04),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
        txt(ax, (x0+x1)/2, yb-0.11, lbl, fontsize=8.5, color=color)

    brace(bx0,  xs_r, by0, '4 static channels', TEAL)
    brace(xd_l, xd_r, by0, 'L=4  Δ²D channels', PURPLE)
    brace(xr_l, xr_r, by0, 'Rain_t', YELLOW)

    txt(ax, bx0 + ch_w/2, by0+bh+0.09,
        'x  =  [static(4) | Δ²D_window(L) | R_t(1)]     shape: (B, 4+L+1, P, P)',
        fontsize=9.5, color=WHITE)

    # ── arrow: channels → TFNO ────────────────────────────────────────────────
    a1_x = bx0 + ch_w + 0.01
    arrow(ax, a1_x, by0+bh/2, a1_x+0.025, by0+bh/2, color=WHITE, lw=1.8)

    # ── TFNO box ──────────────────────────────────────────────────────────────
    tfno_x = a1_x + 0.035
    tfno_w = 0.155
    box(ax, tfno_x, by0-0.09, tfno_w, bh+0.18, '#1a2a4a',
        'Tucker-FNO',
        'Fourier layers × n_layers\nhidden_dim channels\nmodes1×modes2 freqs\n+ domain padding',
        fontsize=9)

    # ── arrow: TFNO → output ──────────────────────────────────────────────────
    a2_x = tfno_x + tfno_w + 0.01
    arrow(ax, a2_x, by0+bh/2, a2_x+0.025, by0+bh/2, color=GREEN, lw=1.8)

    # ── output box ────────────────────────────────────────────────────────────
    out_x = a2_x + 0.035
    out_w = 0.085
    box(ax, out_x, by0+0.02, out_w, bh-0.04, GREEN,
        'Δ²D̂_{t+1}', '(B,1,P,P)', fontsize=9)

    # ── arrow: output → reconstruction ────────────────────────────────────────
    a3_x = out_x + out_w + 0.01
    arrow(ax, a3_x, by0+bh/2, a3_x+0.025, by0+bh/2, color=ORANGE, lw=1.8)

    # ── reconstruction box ────────────────────────────────────────────────────
    rb_x = a3_x + 0.035
    rb_w = 0.185
    box(ax, rb_x, by0-0.07, rb_w, bh+0.14, '#2a1a0a',
        'Depth Reconstruction',
        'D̂_{t+1} = Δ²D̂_{t+1} + 2·D_t − D_{t-1}\nclamp(min=0)  →  physical validity',
        fontsize=9)

    # anchors label
    txt(ax, rb_x + rb_w/2, by0 - 0.15,
        'Anchors  D_{t-1}, D_t  provided by dataset',
        fontsize=8.5, color=ORANGE)

    # ── loss annotation (above TFNO) ──────────────────────────────────────────
    lx = tfno_x + tfno_w/2
    txt(ax, lx, by0+bh+0.28,
        'Loss = MSE( Δ²D̂_{t+1} , Δ²D_{t+1}^{GT} )   [Δ²D space, land-masked]',
        fontsize=10.5, color=RED)
    ax.annotate('', xy=(lx, by0+bh+0.10),
                xytext=(lx, by0+bh+0.23),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    # ── channel colour legend (bottom) ────────────────────────────────────────
    legend = [(TEAL, 'Static (DEM, Manning, Pervious, Slope)'),
              (PURPLE, 'Δ²D history window  (L frames)'),
              (YELLOW, 'Current rainfall  R_t'),
              (GREEN,  'Predicted  Δ²D̂  (model output)'),
              (ORANGE, 'Reconstructed depth  D̂')]
    lx0 = 0.04
    for i, (col, lbl) in enumerate(legend):
        lxi = lx0 + i*0.19
        p   = FancyBboxPatch((lxi, 0.04), 0.013, 0.05,
                             boxstyle='round,pad=0,rounding_size=0.005',
                             facecolor=col, edgecolor='none', alpha=0.9, zorder=4)
        ax.add_patch(p)
        txt(ax, lxi+0.019, 0.065, lbl, fontsize=8, color=WHITE, ha='left')

    fig.savefig(f'{OUT_DIR}/diagram_2_input_tfno.png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved → {OUT_DIR}/diagram_2_input_tfno.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Autoregressive N-step rollout
# ══════════════════════════════════════════════════════════════════════════════

def fig3():
    fig, ax = _make_ax((17, 8))
    fig.text(0.5, 0.97,
             'Part 3 — Autoregressive N-Step Rollout  (training + inference)',
             ha='center', va='top', fontsize=14, color=WHITE, fontweight='bold')

    N           = 3
    step_colors = [BLUE, PURPLE, TEAL]
    sx0         = 0.03
    sy          = 0.36
    sw, sh      = 0.24, 0.32
    hgap        = 0.055

    for k in range(N):
        xk  = sx0 + k*(sw + hgap)
        col = step_colors[k]

        # step frame
        rect = FancyBboxPatch((xk, sy-0.06), sw, sh+0.12,
                              boxstyle='round,pad=0,rounding_size=0.012',
                              facecolor='#1a1f30', edgecolor=col,
                              linewidth=1.5, alpha=0.75, zorder=2)
        ax.add_patch(rect)
        txt(ax, xk+sw/2, sy+sh+0.09, f'Step  k = {k+1}',
            fontsize=11, color=col, fontweight='bold')

        # mini channel strip
        n_ch    = 9
        ci_w    = sw * 0.86 / n_ch
        ci_x0   = xk + sw*0.07
        ci_y    = sy + sh - 0.09
        ci_cols  = [TEAL]*4 + [PURPLE]*4 + [YELLOW]
        ci_lbls  = ['S']*4 + [f'Δ²{j}' for j in range(4)] + ['R']
        for j, (cc, cl) in enumerate(zip(ci_cols, ci_lbls)):
            cx = ci_x0 + j*(ci_w + 0.003)
            p  = FancyBboxPatch((cx, ci_y), ci_w, 0.10,
                                boxstyle='round,pad=0,rounding_size=0.005',
                                facecolor=cc, edgecolor='none', alpha=0.88, zorder=4)
            ax.add_patch(p)
            txt(ax, cx+ci_w/2, ci_y+0.05, cl, fontsize=7, color=WHITE)

        txt(ax, xk+sw/2, ci_y-0.06, 'x  →  TFNO', fontsize=9.5, color=WHITE)

        # Δ²D output box
        pd_y = sy + 0.05
        box(ax, xk+sw*0.22, pd_y, sw*0.56, 0.10, GREEN,
            'Δ²D̂', f't+{k+1}', fontsize=9)
        arrow(ax, xk+sw/2, ci_y-0.02, xk+sw/2, pd_y+0.10, color=GREEN, lw=1.4)

        # reconstruction box
        rec_y = sy - 0.06 + 0.01
        box(ax, xk+sw*0.12, rec_y, sw*0.76, 0.10, ORANGE,
            f'D̂_{{t+{k+1}}}',
            'Δ²D̂ + 2·D_curr − D_prev  |  clamp≥0',
            fontsize=8.5)
        arrow(ax, xk+sw/2, pd_y, xk+sw/2, rec_y+0.10, color=ORANGE, lw=1.4)

        # loss badge
        lbadge_y = rec_y - 0.13
        circle = plt.Circle((xk+sw/2, lbadge_y+0.04), 0.032,
                             color=RED, alpha=0.88, zorder=4)
        ax.add_patch(circle)
        txt(ax, xk+sw/2, lbadge_y+0.04, f'L{k+1}', fontsize=9, color=WHITE,
            fontweight='bold')
        arrow(ax, xk+sw/2, rec_y, xk+sw/2, lbadge_y+0.072, color=RED, lw=1.2)

        # state-transfer arrows to next step
        if k < N - 1:
            xn = xk + sw + hgap
            # Δ²D window slide (top)
            arrow(ax, xk+sw, sy+sh+0.01, xn, sy+sh+0.01, color=PURPLE, lw=1.4)
            txt(ax, xk+sw+hgap/2, sy+sh+0.055,
                'slide Δ²D window', fontsize=8, color=PURPLE)
            # depth anchors (middle)
            arrow(ax, xk+sw, sy+0.08, xn, sy+0.08, color=ORANGE, lw=1.4)
            txt(ax, xk+sw+hgap/2, sy+0.01,
                'D_prev ← D_curr\nD_curr ← D̂_{t+k+1}', fontsize=7.5, color=ORANGE)
            # rain (below channel strip)
            arrow(ax, xk+sw, ci_y+0.04, xn, ci_y+0.04, color=YELLOW, lw=1.4)
            txt(ax, xk+sw+hgap/2, ci_y+0.105,
                'R_{t+k+1}', fontsize=8, color=YELLOW)

    # total loss
    tx = sx0 + N*(sw+hgap) - hgap + 0.03
    txt(ax, tx+0.01, sy+0.12,
        'Total loss\n= Σ L_k\n  k=1…N',
        fontsize=10, color=RED, ha='left', fontweight='bold')
    ax.annotate('', xy=(tx-0.005, sy+0.12),
                xytext=(tx-0.025, sy+0.12),
                arrowprops=dict(arrowstyle='<-', color=RED, lw=1.3))

    # BPTT note
    txt(ax, tx+0.01, sy-0.08,
        'Truncated BPTT\nDetach state between steps\n→ O(1) memory per rollout',
        fontsize=9, color=GREY, ha='left',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#1a1f30',
                  edgecolor=GREY, alpha=0.75))

    # physics loss note
    txt(ax, 0.84, 0.20,
        'Physics loss  (λ_phys > 0)\n\n'
        'If R_t ≈ 0  and  D̂_{t+1} > D_t\n'
        '→ penalise  relu(D̂_{t+1} − D_t)\n\n'
        'Enforces drainage during dry periods',
        fontsize=9, color=WHITE, ha='center',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#251505',
                  edgecolor=ORANGE, alpha=0.90))

    # legend
    legend = [
        (TEAL,   'Static (DEM, Manning, Pervious, Slope)'),
        (PURPLE, 'Δ²D history window  (L frames)'),
        (YELLOW, 'Current rainfall  R_t'),
        (GREEN,  'Predicted  Δ²D̂  (model output)'),
        (ORANGE, 'Reconstructed depth  D̂'),
        (RED,    'MSE loss in Δ²D space  (land-masked)'),
    ]
    lx0 = 0.02
    for i, (col, lbl) in enumerate(legend):
        lx = lx0 + i*0.163
        p  = FancyBboxPatch((lx, 0.03), 0.013, 0.05,
                            boxstyle='round,pad=0,rounding_size=0.005',
                            facecolor=col, edgecolor='none', alpha=0.9, zorder=4)
        ax.add_patch(p)
        txt(ax, lx+0.02, 0.055, lbl, fontsize=8, color=WHITE, ha='left')

    fig.savefig(f'{OUT_DIR}/diagram_3_rollout.png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved → {OUT_DIR}/diagram_3_rollout.png')


# ── run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
