"""Conceptual layer/token dependency figure, not measured experiment data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
target = Path(__file__).resolve().parent / 'figures'
target.mkdir(exist_ok=True)
grid = np.zeros((8,6))
grid[4:, :2] = 1
grid[:,2:] = 2
fig, ax = plt.subplots(figsize=(9,3.8),layout='constrained')
colors=['#dce6eb','#20859c','#d88c47']
ax.imshow(grid,origin='lower',cmap=ListedColormap(colors),vmin=0,vmax=2,aspect='auto')
ax.set_xticks(range(6),labels=range(1,7))
ax.set_yticks(range(8),labels=range(8))
ax.set_xlabel('Replay transition (example window W = 6)')
ax.set_ylabel('Decoder layer')
ax.hlines(3.5,-.5,1.5,color='#173a49',linewidth=1.5)
ax.axvline(1.5,color='#6b4725',linewidth=1.5)
ax.text(0.5,6.0,'Replay from\nclean cut c = 4',ha='center',va='center',color='white',fontsize=11)
ax.text(0.5,1.6,'Retain valid\nlower-layer KV',ha='center',va='center',color='#193c4a',fontsize=10)
ax.text(3.5,4,'All-layer replay after\nchanged output at transition 2',ha='center',va='center',color='#2e251d',fontsize=11)
ax.set_title('Conceptual example: token divergence limits safe reuse',loc='left',weight='bold',fontsize=12,pad=12)
ax.legend(handles=[Patch(color=colors[0],label='Retained'),Patch(color=colors[1],label='Upper-layer replay'),Patch(color=colors[2],label='Full replay')],loc='upper center',bbox_to_anchor=(.5,-.21),ncol=3,frameon=False)
fig.savefig(target/'recovery_frontier.png',dpi=180,bbox_inches='tight')
plt.close(fig)
