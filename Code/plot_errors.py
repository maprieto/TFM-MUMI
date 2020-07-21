import numpy as np
import matplotlib.pylab as plt
import sys
from scipy import stats

probIdStr = 'rigid-rigid_fluid'

h=1./np.array([50.,100.,200.,400.])
error=np.array([9.461108877125268, 2.3439457246305806, 0.6576513648272083, 0.15344530778140894])

slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(h), np.log(error))

fig = plt.figure()
fig.subplots_adjust(wspace=0.2, hspace=1)
textAlignX = 0.05; textAlignY = 0.75; fontSize = 12
    
ax = fig.gca()
ax.loglog(h, error, marker="o")
ax.loglog(h, np.exp(intercept + slope * np.log(h)), color='red')
ax.set(xlabel='$\Delta t$', ylabel='$L^2$-relative error (%)', title='$CFL=1$')
ax.text(textAlignX, textAlignY, 'error=%g+%g*h' % (intercept, slope), transform=ax.transAxes, fontsize=fontSize)

plt.show()

#fig.savefig("figures/%s_error.png")

print('Done')
