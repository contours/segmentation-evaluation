import matplotlib.pyplot as plt
import numpy as np
import itertools as it

from brewer2mpl import qualitative, sequential

from helpers import mean_std_n

# Some nice colors
ALMOST_BLACK = '#262626'
DARK_GRAY = np.array([float(200)/float(255)]*3)
LIGHT_GRAY = np.array([float(248)/float(255)]*3)

def title_to_filename(title):
    return title.lower().replace(' ', '-')

def plot(x, y, xlabel, ylabel):
    fit = np.polyfit(x, y, 1)
    plt.plot(x, y, marker='.', linestyle='None', color=ALMOST_BLACK)
    plt.plot(x, np.polyval(fit, x), linestyle='--', color=DARK_GRAY)
    plt.tick_params(colors=ALMOST_BLACK)
    plt.xlabel(xlabel, color=ALMOST_BLACK)
    plt.ylabel(ylabel, color=ALMOST_BLACK)
    for spine in plt.gca().spines.values():
        spine.set_color(ALMOST_BLACK)
    plt.savefig(title_to_filename(xlabel+' vs '+ylabel) + '.svg')
    plt.show()

# Draw a Cleveland-style dot plot with possibly multiple values per label
def dot_plot(values, labels, title, ytitle=None, sort=True,
             value_labels=None, minv=None, maxv=None, draw_mean_std=False,
             color_offset=0, draw_line=None, errors=None, ns=None,
             scale_w=1.0, scale_h=1.0):

    if len(values) == 0: return

    if isinstance(values[0], (list, tuple)):
        draw_mean_std = False
    else:
        values = [ values ]

    draw_legend = True

    if not value_labels:
        value_labels = [''] * len(labels)
        draw_legend = False

    if len(values) > 1:
        colors = qualitative.Set2[len(values) + color_offset].mpl_colors[color_offset:]
    else:
        colors = [ ALMOST_BLACK ]

    if errors or ns:
        sort = False

    if sort:
        # order by mean of the multiple values
        means = [ np.mean(np.array(list(v), dtype=float)) for v in zip(*values) ]
        ordered = list(zip(*sorted(zip(*[means] + [labels] + values))))
        labels, values = ordered[1], ordered[2:]

    y = range(len(labels))

    if minv is None:
        minv = float(min(it.chain(*values)))
    if maxv is None:
        maxv = float(max(it.chain(*values)))
    width = maxv - minv
    left = minv - 0.05 * width
    right = maxv + 0.05 * width

    plt.xlim(left, right)
    plt.ylim(-1, len(labels))

    plt.yticks(y, labels);
    plt.tick_params(axis='x', direction='in', labeltop='on', colors=ALMOST_BLACK)
    plt.tick_params(axis='y', left='off', right='off', colors=ALMOST_BLACK)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.hlines(y, left, right, linestyles='dotted',
               linewidths=0.5, colors=ALMOST_BLACK)
    plt.xlabel(title, color=ALMOST_BLACK)
    if ytitle: plt.ylabel(ytitle, color=ALMOST_BLACK)

    for v, color, label in zip(values, colors, value_labels):
        if errors is not None:
            plt.errorbar(v, y, xerr=errors, color=ALMOST_BLACK, linestyle='None')
        plt.plot(v, y, 'o', color=color, alpha=0.75, label=label)

    if ns is not None:
        for n,yval in zip(ns,y):
            plt.annotate('n=%s' % n, (right, yval), textcoords='offset points', xytext=(4,-2))

    if draw_line is not None:
        if draw_mean_std:
            raise Exception('Cannot use draw_line and draw_mean_std together')
        plt.vlines(draw_line, 0, len(labels)-1, colors=[ALMOST_BLACK])

    if draw_mean_std:
        mean, std, n = mean_std_n(values)
        plt.vlines(mean, 0, len(labels)-1, colors=[ALMOST_BLACK])
        plt.vlines([mean-std, mean+std], 0, len(labels)-1, colors=[DARK_GRAY])

    if draw_legend:
        leg = plt.legend(frameon=True, numpoints=1, loc='center',
                         bbox_to_anchor=(1,0.5))
        plt.setp(leg.get_frame(), facecolor=LIGHT_GRAY, edgecolor='none')
        for t in leg.texts: t.set_color(ALMOST_BLACK)

    w,h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches([w*scale_w, h*scale_h])
        
    plt.savefig(title_to_filename(title) + '.svg')
    plt.show()

# Draw a box-and-whisker plot
def box_plot(values, labels, title):
    values = [ np.array(v, dtype=float) for v in values ]
    medians = [ np.median(v) for v in values ]
    medians, values, labels = zip(*sorted(zip(medians, values, labels), key=lambda x: x[0]))
    bp = plt.boxplot(values, sym='', vert=False, widths=0.7)

    plt.setp(bp['boxes'], color=ALMOST_BLACK, linewidth=0.5)
    plt.setp(bp['whiskers'], color=ALMOST_BLACK, linewidth=0.5)
    plt.setp(bp['medians'], color='#e41a1c', linewidth=0.5)

    y = range(1, len(values)+1)
    plt.ylim(0, len(values)+1)

    plt.yticks(y, labels)
    plt.tick_params(axis='x', direction='in', colors=ALMOST_BLACK)
    plt.tick_params(axis='y', left='off', right='off', colors=ALMOST_BLACK)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xlabel(title, color=ALMOST_BLACK)

    plt.savefig(title_to_filename(title) + '.svg')
    plt.show()

def bar_chart(values, labels, value_labels, title, scale_w=1.0, scale_h=1.0,
              legend_x=1.0):
    bottom = np.arange(len(labels)) * 0.4 + 0.1
    height = 0.2
    data = np.array(values)
    left = np.vstack((np.zeros((data.shape[1],), dtype=data.dtype), np.cumsum(data, axis=0)[:-1]))
    colors = sequential.Greys[len(data)].mpl_colors
    rectangles = [ plt.barh(bottom, d, height, left=l, color=c, edgecolor='None')[0]
                   for d,c,l in zip(data, colors, left) ]
    plt.yticks(bottom+height/2., labels)
    plt.tick_params(axis='x', direction='in', labeltop='on', colors=ALMOST_BLACK)
    plt.tick_params(axis='y', left='off', right='off', colors=ALMOST_BLACK)
    plt.xlabel(title, color=ALMOST_BLACK)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    w,h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches([w*scale_w, h*scale_h])

    leg = plt.legend(rectangles, value_labels, frameon=False,
                     loc='lower left', bbox_to_anchor=(legend_x,0.5),
                     fontsize='medium')
    for t in leg.texts: t.set_color(ALMOST_BLACK)

    plt.savefig(title_to_filename(title) + '.svg')
    plt.show()

def histogram(values, bins, title, ytitle=None, scale_w=1.0, scale_h=1.0):
    values = np.array(values, dtype=float)
    plt.hist(values, bins=bins, color=DARK_GRAY, edgecolor=DARK_GRAY)
    plt.xlabel(title, color=ALMOST_BLACK)
    if ytitle: plt.ylabel(ytitle, color=ALMOST_BLACK)

    w,h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches([w*scale_w, h*scale_h])

    plt.savefig(title_to_filename(title) + '.svg')
    plt.show()

def heatmap(values, row_labels, column_labels, title, scale_w=0.8, scale_h=2.0):
    array = np.array(values, dtype=float)

    assert(array.shape[1] == len(column_labels))
    assert(array.shape[0] == len(row_labels))

    # Thanks https://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
    fig, ax = plt.subplots()
    collection = ax.pcolor(array, vmin=0, vmax=1, cmap=plt.cm.binary, alpha=0.8)
    fig.colorbar(collection)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(array.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(array.shape[0])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    plt.xlabel(title, color=ALMOST_BLACK)
    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)

    w,h = fig.get_size_inches()
    plt.gcf().set_size_inches([w*scale_w, h*scale_h])
    plt.ylim(len(row_labels), 0)
    plt.tick_params(axis='x', top='off', bottom='off', colors=ALMOST_BLACK)
    plt.tick_params(axis='y', left='off', right='off', colors=ALMOST_BLACK)

    plt.savefig(title_to_filename(title) + '.svg')
    plt.show()

def table(rows, header_rows=1):
    thead = '<thead>%s</thead>' % ''.join([ '<tr>%s</tr>' % ''.join(
        [ '<th>%s</th>' % value for value in row ])
            for row in rows[:header_rows] ])
    tbody = '<tbody>%s</tbody>' % ''.join([ '<tr>%s</tr>' % ''.join(
        [ '<td>%s</td>' % value for value in row ])
            for row in rows[header_rows:] ])
    sums = [ sum(values) if (type(values[0]) == int)
             else '' for values in zip(*rows[header_rows:]) ]
    tfoot = '<tfoot><tr>%s</tr></tfoot>' % ''.join(
        [ '<th>%s</th>' % s for s in sums ])
    return '<table>%s</table>' % ''.join([thead,tbody,tfoot])
