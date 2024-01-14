from sklearn.cluster import KMeans, SpectralClustering, MeanShift, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, \
                            AffinityPropagation, Birch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

np.random.seed = 42
MAX_CLUSTERS = 40
MIN_CLUSTERS = 2

input_options = {
    'all': './data/coffee.csv',
    'phrases': './data/phrases.csv',
    'sentences': './data/sentences.csv',
    'test': './data/Articles_medical.csv'
}
coding_optincs = {
    # 'count': {
    #     'func': CountVectorizer,
    #     'int': {'min_df': (1, 300, 0.01),
    #             # 'max_df': (0.9, 1.0, 0.01)
    #             },
    #     'float': {
    #         # 'min_df': (0.01, 0.1, 0.01),
    #         'max_df': (0.8, 1.0, 0.01)
    #     },
    #     'categorical': {
    #         #'ngram_range': [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
    #         'analyzer': ['word', 'char', 'char_wb']
    #     }
    # },
    'tfidf': {
        'func': TfidfVectorizer,
        'int': {'min_df': (5, 100, 1),
                #'max_df': (0.9, 1.0, 0.01)
                },
        'float': {
            #'min_df': (0.01, 0.1, 0.01),
            'max_df': (0.8, 0.9, 0.002)
            },
        'categorical': {
            #'ngram_range': [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
            'analyzer': ['word'], #, 'char', 'char_wb'],
            'norm': ('l1', 'l2', None),
            'use_idf': (True, False),
            'smooth_idf': (True, False),
            'sublinear_tf': (True, False),
            }
    },
}
model_options = {
    'KMeans': {
        'func': KMeans,
        'int': {'n_clusters': (MIN_CLUSTERS, MAX_CLUSTERS),
                'max_iter': (300, 1000),
                'random_state': (42, 42)},  #
        'float': {},
        'categorical': {'init': ('k-means++', 'random'),
                        'n_init': ('auto', ),
                        'algorithm': ['lloyd', 'elkan']}
    },
    # очень долго на 5к с word - потому что размерность высокая очень
    # 'SpectralClustering': {
    #     'func': SpectralClustering,
    #     'int': {'n_clusters': (MIN_CLUSTERS, MAX_CLUSTERS),
    #             'n_neighbors': (2, 20),
    #             'degree': (1, 10),
    #             'random_state': (42, 42)},  #
    #     'float': {
    #         'gamma': (0.1, 100, 0.1),
    #         'coef0': (0.1, 2, 0.01)
    #     },
    #     'categorical': {'eigen_solver': ('arpack', 'lobpcg'),
    #                     'affinity': ('nearest_neighbors', 'rbf'),
    #                     'assign_labels': ['kmeans',  'cluster_qr'] # 'discretize', - Ошибка SVD did not converge
    #     }
    # },
    # 'MeanShift': {
    #     'func': MeanShift,
    #     'int': {},
    #     'float': {'bandwidth': (2, 100, 0.1)},
    #     'categorical': {'cluster_all': (True, False),
    #                     'bin_seeding': (True, False)}
    # },
    'AgglomerativeClustering': {
        'func': AgglomerativeClustering,
        'int': {'n_clusters': (MIN_CLUSTERS, MAX_CLUSTERS)},
        'float': {},
        'categorical': {'linkage': ('complete', 'average', 'single'),
                        'metric': ('cityblock',  'euclidean', 'l1', 'l2', 'manhattan')} # 'cosine',
    },
    'AgglomerativeClustering_ward': {
        'func': AgglomerativeClustering,
        'int': {'n_clusters': (MIN_CLUSTERS, MAX_CLUSTERS)},
        'float': {},
        'categorical': {'linkage': ('ward', ),
                        'metric': ('euclidean', )}
        },
    # Дают 2 кластера, неравномерных
    # 'DBSCAN': {
    #     'func': DBSCAN,
    #     'int': {'p': (1, 10)},
    #     'float': {'eps': (0.001, 10, 0.01)},
    #     'categorical': {'metric': ('cityblock',  'euclidean', 'l1', 'l2', 'manhattan'),
    #                     'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}
    # },
    # 'HDBSCAN': {
    #     'func': HDBSCAN,
    #     'int': {'min_cluster_size': (5, 100),
    #             'min_samples': (5, 100)},
    #     'float': {'cluster_selection_epsilon': (0, 10, 0.1),
    #               'alpha': (0.1, 100, 0.1)},
    #     'categorical': {'metric': ('cityblock', 'euclidean', 'l1', 'l2', 'manhattan'),
    #                     'algorithm': ('auto', 'balltree', 'kdtree', 'brute'),
    #                     'cluster_selection_method': ('eom', 'leaf'),
    #                     'store_centers': (None, 'centroid', 'medoid', 'both')}
    # },
    'OPTICS': {
        'func': OPTICS,
        'int': {'p': (1, 10),
                'min_samples': (5, 100)},
        'float': {'eps': (0.001, 10, 0.1)},
        'categorical': {'metric': ('cityblock',  'euclidean', 'l1', 'l2', 'manhattan'),
                        'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                        'cluster_method': ('xi', 'dbscan')}
    },
    'AffinityPropagation': {
        'func': AffinityPropagation,
        'int': {},
        'float': {'damping': (0.5, 0.999, 0.01)},
        'categorical': {}
    },
    'Birch': {
        'func': Birch,
        'int': {'n_clusters': (MIN_CLUSTERS, MAX_CLUSTERS),
                'branching_factor': (5, 100)},
        'float': {'threshold': (0.01, 5, 0.1)},
        'categorical': {}
    }

}

