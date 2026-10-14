import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
for ax, mode, title in ((axes[0], 'classic', 'classic —— 匀速巡逻 + 连续自转（改动前的行为）'),
                        (axes[1], 'gyro', 'gyro —— 巡逻 + 静止小陀螺甩头（新增，橙色=小陀螺期间）')):
    d = np.genfromtxt('/tmp/curve_%s.csv' % mode, delimiter=',', dtype=str)
    t = d[:, 0].astype(float); w = d[:, 1].astype(float); m = d[:, 2]
    ax.plot(t, w, lw=1.2, color='#1f77b4', label='靶车自转 ω')
    gy = m == 'gyro'
    if gy.any():
        ax.fill_between(t, 0, np.where(gy, w, 0), color='#ff7f0e', alpha=0.4, label='小陀螺（车体静止）')
    ax.axhline(7.02, ls='--', lw=0.8, color='gray')
    ax.text(0.2, 7.3, '巡逻自转上限 7.02 rad/s', fontsize=7, color='gray')
    ax.set_ylabel('自转角速度 (rad/s)')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
axes[1].set_xlabel('时间 (s)')
plt.tight_layout()
plt.savefig('/tmp/car_motion_curves.png', dpi=110)
print('OK')
