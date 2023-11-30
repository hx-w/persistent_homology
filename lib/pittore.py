# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_complex(origin_data, rips_complex, bound=[-6,8,-6,6]):
    plt.clf()
    plt.axis(bound) #bound = [x1, x2, y1, y2]
    plt.scatter(origin_data[:,0], origin_data[:,1]) #plotting just for clarity
    for i, txt in enumerate(origin_data):
        plt.annotate(i, (origin_data[i][0]+0.05, origin_data[i][1])) #add labels

    #add lines for edges
    for edge in [e for e in rips_complex if len(e)==2]:
        #print(edge)
        pt1,pt2 = [origin_data[pt] for pt in [n for n in edge]]
        #plt.gca().add_line(plt.Line2D(pt1,pt2))
        
        line = plt.Polygon([pt1, pt2], closed=None, fill=None, edgecolor='r')
        plt.gca().add_patch(line)

    #add triangles
    for triangle in [t for t in rips_complex if len(t)==3]:
        pt1,pt2,pt3 = [origin_data[pt] for pt in [n for n in triangle]]
        line = plt.Polygon([pt1, pt2, pt3], closed=False, color="blue",alpha=0.3, fill=True, edgecolor=None)
        plt.gca().add_patch(line)
    plt.show()

def graph_barcode(persistence, homology_group=0):
    #this function just produces the barcode graph for each homology group
    xstart = [s[1][0] for s in persistence if s[0] == homology_group]
    xstop = [s[1][1] for s in persistence if s[0] == homology_group]
    y = [0.1 * x + 0.2 for x in range(len(xstart))]
    plt.hlines(y, xstart, xstop, color='b', lw=3)
    #Setup the plot
    ax = plt.gca()
    plt.ylim(0, max(y) + 0.1)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.xlabel('epsilon')
    plt.ylabel("Betti dim %s" % (homology_group,))
    plt.show()

def graph_diagram(persistence):
    sns.set()

    pd_data = np.array([[element[0], *element[1]] for element in persistence])
    sns.scatterplot(
        data={'Betti': pd_data[:, 0], 'Birth': pd_data[:, 1], 'Death': pd_data[:, 2]},
        x='Birth', y='Death', hue='Betti', palette='Set1'
    )

    max_ = np.max(pd_data[:, 1:])

    plt.plot([0, max_ + 0.1], [0, max_ + 0.1], color='grey', linestyle='dashed')

    plt.show()