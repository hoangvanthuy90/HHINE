import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import check_random_state

#path = "data/wordnet_mammal_hypernyms.tsv"
path = "data/mammals_sample.tsv"



def load_data(file_path, delim="\t"):
    data = []
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        for i, line in enumerate(reader):
            data.append(line)
    return data
mammal_relations = load_data(path)
mammal_relations[:5]

mammal = pd.read_csv('mammal_closure.csv')
print('Total unique nodes: ', len(np.unique(list(mammal.id1.values) + list(mammal.id2.values))))
mammal_relations = [[mammal.id1[i].split('.')[0], mammal.id2[i].split('.')[0]] for i in range(len(mammal))]
#mammal_relations = [[mammal.id1[i], mammal.id2[i]] for i in range(len(mammal))]
print('Total relations: ', len(mammal_relations))
print('# of (u, u) type relations: ', len([r for r in mammal_relations if r[0]==r[1]]))
print('First ten relations:\n', mammal_relations[:10])
unique_nodes = np.unique([item for sublist in mammal_relations for item in sublist])
def init_embeddings(n, dim, low=-0.001, high=0.001):
    theta_init = np.random.uniform(low, high, size=(n, dim))
    return theta_init
emb = init_embeddings(len(unique_nodes), 2)
emb_dict = dict(zip(unique_nodes, emb))
{k: emb_dict[k] for k in list(emb_dict)[:10]}


def negative_sample(data, u, n_samples):
    positives = [x[1] for x in data if x[0] == u]
    negatives = np.array([x for x in unique_nodes if x not in positives])
    # negatives = np.array([x[1] for x in data if x[1] not in positives])
    random_ix = np.random.permutation(len(negatives))[:n_samples]
    neg_samples = [[u, x] for x in negatives[random_ix]]
    neg_samples.append([u, u])
    return neg_samples


negative_sample(mammal_relations, 'kangaroo', 4)

eps = 1e-5


def partial_d(theta, x):
    alpha = 1 - norm(theta) ** 2
    beta = 1 - norm(x) ** 2
    gamma = 1 + 2 / (alpha * beta + eps) * norm(theta - x) ** 2
    lhs = 4 / (beta * np.sqrt(gamma ** 2 - 1) + eps)
    rhs = 1 / (alpha ** 2 + eps) * (norm(x) ** 2 - 2 * np.inner(theta, x) + 1) * theta - x / (alpha + eps)
    return lhs * rhs


def proj(theta):
    if norm(theta) >= 1:
        theta = theta / norm(theta) - eps
    return theta


def update(u, lr, grad, embeddings, test=False):
    theta = embeddings[u]
    step = 1 / 4 * lr * (1 - norm(theta) ** 2) ** 2 * grad
    embeddings[u] = proj(theta - step)
    if test:
        if norm(proj(theta - step) < norm(theta)):
            print('updating ' + u + ' closer to origin')
        else:
            print('updating ' + u + ' away from origin')
    return


import time

num_neg = 10


def train_poincare(relations, lr=0.01, num_epochs=10):
    for i in range(num_epochs):
        # loss=0
        start = time.time()
        for relation in relations:
            u, v = relation[0], relation[1]
            if u == v:
                continue
            # embedding vectors (theta, x) for relation (u, v)
            theta, x = emb_dict[u], emb_dict[v]
            # embedding vectors v' in sample negative relations (u, v')
            neg_relations = [x[1] for x in negative_sample(relations, u, num_neg)]
            neg_embed = np.array([emb_dict[x] for x in neg_relations])
            # find partial derivatives of poincare distance
            dd_theta = partial_d(theta, x)
            dd_x = partial_d(x, theta)
            # find partial derivatives of loss function
            dloss_theta = -1
            dloss_x = 1
            grad_theta = dloss_theta * dd_theta
            grad_x = dloss_x * dd_x
            update(u, lr, grad_theta, emb_dict)
            update(v, lr, grad_x, emb_dict)
            # find gradients for negative samples
            neg_exp_dist = np.array([np.exp(-poincare_dist(theta, v_prime)) for v_prime in neg_embed])
            Z = neg_exp_dist.sum(axis=0)
            for vprime in neg_relations:
                dd_vprime = partial_d(emb_dict[vprime], theta)
                dd_u = partial_d(theta, emb_dict[vprime])
                dloss_vprime = -np.exp(-poincare_dist(emb_dict[vprime], theta)) / Z
                dloss_u = -np.exp(-poincare_dist(theta, emb_dict[vprime])) / Z
                grad_vprime = dd_vprime * dloss_vprime
                grad_u = dd_u * dloss_u
                update(vprime, lr, grad_vprime, emb_dict)
                update(u, lr, grad_u, emb_dict)
            # loss = loss + np.log(np.exp(-poincare_dist(theta, x))) / Z

        print('COMPLETED EPOCH ', i + 1)
        # print(' LOSS: ', loss)
        print('---------- total seconds: ', time.time() - start)
def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)
train_poincare(mammal_relations, lr=0.01, num_epochs=30)

def positive_ranks(item, relations, embedding_dict):
    theta = embedding_dict[item]
    distances = [poincare_dist(theta, x) for x in np.array(list(embedding_dict.values()))]
    positives = [x[1] for x in relations if x[0] == item]
    keys = list(embedding_dict.keys())
    ranks = [keys[i] for i in np.argsort(distances)]
    pos_ranks = [j for j in range(len(ranks)) if ranks[j] in positives]
    return pos_ranks
#positive_ranks('antelope.n.01', mammal_relations, emb_dict)

# positive ranks = [a, b, c, d] -> avg precision = avg(1/a + 2/b + 3/c + 4/d)
def avg_precision(item, relations, embedding_dict):
    ranks = positive_ranks(item, relations, embedding_dict)
    map_ranks = np.sort(ranks) + np.arange(len(ranks))
    avg_precision = ((np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean())
    return avg_precision

def mean_average_precision(relations, embedding_dict):
    avg_precisions = []
    ranks = []
    for item in list(embedding_dict.keys()):
        if not np.isnan(avg_precision(item, relations, embedding_dict)):
            avg_precisions.append(avg_precision(item, relations, embedding_dict))
        if len(positive_ranks(item, relations, embedding_dict)) != 0:
            ranks = ranks+positive_ranks(item, relations, embedding_dict)
    return [ranks, avg_precisions]

def dist_squared(x, y, axis=None):
    return np.sum((x - y)**2, axis=axis)

def get_subtree(relations, embedding_dict, root_node):
    root_emb = embedding_dict[root_node]
    child_nodes = [x[0] for x in relations if x[1] == root_node]
    child_emb = np.array([embedding_dict[x[0]] for x in relations if x[1] == root_node])
    return [child_nodes, child_emb]
#get_subtree(mammal_relations, emb_dict, 'feline')
#for child in get_subtree(mammal_relations, embedding_dict, 'feline.n.01'):
#    ax.plot([embedding_dict['feline.n.01'][0], child[0]], [embedding_dict['feline.n.01'][1], child[1]], '--', c='black')


def plot_embedding(embedding_dict, label_frac=0.001, plot_frac=0.6, title=None, save_fig=False):
    fig = plt.figure(figsize=(8,8))
    plt.grid('off')
    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])
    plt.axis('off')
    ax = plt.gca()
    embed_vals = np.array(list(embedding_dict.values()))
    #plt.xlim([embed_vals.min(0)[0],embed_vals.max(0)[0]])
    #plt.ylim([embed_vals.min(0)[1],embed_vals.max(0)[1]])
    keys = list(embedding_dict.keys())
    min_dist_2 = label_frac * max(embed_vals.max(axis=0) - embed_vals.min(axis=0)) ** 2
    labeled_vals = np.array([2*embed_vals.max(axis=0)])
    groups = [keys[i] for i in np.argsort(np.linalg.norm(embed_vals, axis=1))][:10]
    #groups.insert(0, 'mammal.n.01')
    for key in groups:
        if np.min(dist_squared(embedding_dict[key], labeled_vals, axis=1)) < min_dist_2:
            continue
        else:
            _ = ax.scatter(embedding_dict[key][0], embedding_dict[key][1], s=40)
            props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.5)
            _ = ax.text(embedding_dict[key][0], embedding_dict[key][1]+0.01, s=key.split('.')[0],
                        size=8, fontsize=10, verticalalignment='top', bbox=props)
            labeled_vals = np.vstack((labeled_vals, embedding_dict[key]))
    n = int(plot_frac*len(embed_vals))
    for i in np.random.permutation(len(embed_vals))[:n]:
        _ = ax.scatter(embed_vals[i][0], embed_vals[i][1],  s=40)
        if np.min(dist_squared(embed_vals[i], labeled_vals, axis=1)) < min_dist_2:
            continue
        else:
            _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i].split('.')[0],
                        size=6, fontsize=8, verticalalignment='top', bbox=props)
            labeled_vals = np.vstack((labeled_vals, embed_vals[i]))
    if title != None:
        plt.title(title, size=20)
    if save_fig:
        plt.savefig('images/poincare_mammals_sample.png')
    print(labeled_vals.shape)


emb_ranks, emb_precisions = mean_average_precision(mammal_relations, emb_dict)
print('MEAN RANK: ', np.mean(emb_ranks))
print('MAP SCORE: ', np.mean(np.nan_to_num(emb_precisions)))


plot_embedding(emb_dict, plot_frac=1, label_frac=0.001,
               title='Poincaré Embeddings: Mammals Sample', save_fig=True)
