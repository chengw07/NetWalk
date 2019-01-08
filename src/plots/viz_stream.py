"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pylab as pylab



params = {'font.family': 'sans-serif', 'font.serif': 'Times'}
pylab.rcParams.update(params)



def viz_stream(rm_edges, fig, row, col, id, membership_path="./tmp/membership.txt", embedding_path="./tmp/embedding.txt"):
    """
    Visualizing karate graph for dynamical graph embedding demo
    :param rm_edges: removed edges
    :param fig: figure
    :param row: subplot row number
    :param col: subplot col number
    :param id: subplot id
    :param membership_path: node class in karate graph
    :param embedding_path: node embedding results
    :return: no return
    """
    G = nx.karate_club_graph()

    membership = {}
    with open(membership_path, 'r') as member:
        for idx, line in enumerate(member):
            membership[idx] = int(line.strip())

    # https://networkx.github.io/documentation/development/_modules/networkx/drawing/layout.html
    pos = nx.fruchterman_reingold_layout(G, seed=12)
    #pos = nx.spring_layout(G)

    # http://matplotlib.org/mpl_examples/color/named_colors.pdf
    colors = ['orange', 'orangered', 'darkturquoise', 'goldenrod', 'dodgerblue']


    ff = fig.add_subplot(row, col, 2*(id-1)+1)
    ff.patch.set_visible(True)
    ff.axis('off')

    G.remove_edges_from(rm_edges)

    count = 0
    for com in set(membership.values()):
        count += 1
        list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=150, linewidths=0, node_color=colors[count])
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='k')#, font=font)
    nx.draw_networkx_edges(G, pos, alpha=0.5)


    fig.add_subplot(row, col, 2*(id-1)+2)
    ff.patch.set_visible(True)
    ff.axis('off')


    embedding = {}
    with open(embedding_path, 'r') as member:
        # member.readline()
        for line in member:
            res = line.strip().split()
            embedding[int(res[0])] = [float(res[1]), float(res[2])]
    count = 0
    for com in set(membership.values()):
        count += 1
        list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
        nx.draw_networkx_nodes(G, embedding, list_nodes, node_size=150, linewidths=0., node_color=colors[count])
    nx.draw_networkx_labels(G, embedding, font_size=9, font_color='k')#, font=font)


    # plt.savefig('/Users/wenchao/Drive/Research/Projects/StreamAnomaly/figs/fig3-embed3.eps', bbox_inches='tight', pad_inches=0)

    # pp = PdfPages('/Users/Wenchao/Google Drive/AnomalyDetection/figs/fig3-embed2.pdf')
    # plt.savefig(pp, format='pdf', bbox_inches='tight', pad_inches=0)
    # pp.close()


if __name__ == "__main__":
    # G = nx.path_graph(8)
    # nx.draw(G)
    # plt.show()

    membership_path = "../tmp/membership.txt"
    embedding_path = "../tmp/embedding.txt"
    viz_stream(membership_path, embedding_path)


