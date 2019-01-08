"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pylab as pylab



params = {'font.family': 'sans-serif', 'font.serif': 'Times'}
pylab.rcParams.update(params)

# font = {'fontname'   : 'Times New Roman',
#         'color'      : 'k',
#         'fontweight' : 'bold',
#         'fontsize'   : 14}


def viz(membership_path="./tmp/membership.txt", embedding_path="./tmp/embedding.txt"):
    G = nx.karate_club_graph()

    membership = {}
    with open(membership_path, 'r') as member:
        for idx, line in enumerate(member):
            membership[idx] = int(line.strip())

    # https://networkx.github.io/documentation/development/_modules/networkx/drawing/layout.html
    pos = nx.fruchterman_reingold_layout(G)
    #pos = nx.spring_layout(G)

    # http://matplotlib.org/mpl_examples/color/named_colors.pdf
    colors = ['orange', 'orangered', 'darkturquoise', 'goldenrod', 'dodgerblue']

    fig = plt.figure(figsize=(8, 6))


    # fig, ax  = plt.subplots()
    # fig.patch.set_visible(False)
    # ax.axis('off')

    ff = fig.add_subplot(2,1,1)
    ff.patch.set_visible(True)
    ff.axis('off')



    count = 0
    for com in set(membership.values()):
        count += 1
        list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=200, linewidths=0, node_color=colors[count])
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='k')#, font=font)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    #plt.show()


    #fig, ax = plt.subplots()

    fig.add_subplot(2,1,2)
    ff.patch.set_visible(True)
    ff.axis('off')


    # ax.get_yaxis().set_tick_params(which='major', direction='out')
    # ax.get_xaxis().set_tick_params(which='major', direction='out')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

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
        nx.draw_networkx_nodes(G, embedding, list_nodes, node_size=200, linewidths=0., node_color=colors[count])
    nx.draw_networkx_labels(G, embedding, font_size=9, font_color='k')#, font=font)
    plt.show()

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
    viz(membership_path, embedding_path)


