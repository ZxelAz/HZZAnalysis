"""Shared configuration for the HZZ STXS categorisation stage.

STXS categorisation dictionaries, merge helpers, and run-2 category mapping.
Consumed by the BDT, DNN, GATO, and CutBased per-method config shims under
``Categorisation/<method>/functions/config.py``.

Method-specific training hyperparameters (e.g. XGBoost params, Optuna search
space) do NOT live here — they stay in the method's own config.py shim.
"""

# STXS Stage 0 categories
STXS_STAGE_0_DICT = {
    0: 'UNKNOWN',
    10: 'GG2H_FWDH',
    11: 'GG2H',
    20: 'VBF_FWDH',
    21: 'VBF',
    22: 'VH2HQQ_FWDH',
    23: 'VH2HQQ',
    30: 'QQ2HLNU_FWDH',
    31: 'QQ2HLNU',
    40: 'QQ2HLL_FWDH',
    41: 'QQ2HLL',
    50: 'GG2HLL_FWDH',
    51: 'GG2HLL',
    60: 'TTH_FWDH',
    61: 'TTH',
    70: 'BBH_FWDH',
    71: 'BBH',
    80: 'TH_FWDH',
    81: 'TH'
}

# STXS Stage 1.2 categories
STXS_STAGE_1_2_DICT = {
    0: 'UNKNOWN',
    # Gluon fusion
    100: 'GG2H_FWDH',
    101: 'GG2H_PTH_200_300',
    102: 'GG2H_PTH_300_450',
    103: 'GG2H_PTH_450_650',
    104: 'GG2H_PTH_GT650',
    105: 'GG2H_0J_PTH_0_10',
    106: 'GG2H_0J_PTH_GT10',
    107: 'GG2H_1J_PTH_0_60',
    108: 'GG2H_1J_PTH_60_120',
    109: 'GG2H_1J_PTH_120_200',
    110: 'GG2H_GE2J_MJJ_0_350_PTH_0_60',
    111: 'GG2H_GE2J_MJJ_0_350_PTH_60_120',
    112: 'GG2H_GE2J_MJJ_0_350_PTH_120_200',
    113: 'GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    114: 'GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
    115: 'GG2H_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    116: 'GG2H_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',
    # VBF
    200: 'QQ2HQQ_FWDH',
    201: 'QQ2HQQ_0J',
    202: 'QQ2HQQ_1J',
    203: 'QQ2HQQ_GE2J_MJJ_0_60',
    204: 'QQ2HQQ_GE2J_MJJ_60_120',
    205: 'QQ2HQQ_GE2J_MJJ_120_350',
    206: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',
    207: 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    208: 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
    209: 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    210: 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',
    # qq -> WH
    300: 'QQ2HLNU_FWDH',
    301: 'QQ2HLNU_PTV_0_75',
    302: 'QQ2HLNU_PTV_75_150',
    303: 'QQ2HLNU_PTV_150_250_0J',
    304: 'QQ2HLNU_PTV_150_250_GE1J',
    305: 'QQ2HLNU_PTV_GT250',
    # qq -> ZH
    400: 'QQ2HLL_FWDH',
    401: 'QQ2HLL_PTV_0_75',
    402: 'QQ2HLL_PTV_75_150',
    403: 'QQ2HLL_PTV_150_250_0J',
    404: 'QQ2HLL_PTV_150_250_GE1J',
    405: 'QQ2HLL_PTV_GT250',
    # gg -> ZH
    500: 'GG2HLL_FWDH',
    501: 'GG2HLL_PTV_0_75',
    502: 'GG2HLL_PTV_75_150',
    503: 'GG2HLL_PTV_150_250_0J',
    504: 'GG2HLL_PTV_150_250_GE1J',
    505: 'GG2HLL_PTV_GT250',
    # ttH
    600: 'TTH_FWDH',
    601: 'TTH_PTH_0_60',
    602: 'TTH_PTH_60_120',
    603: 'TTH_PTH_120_200',
    604: 'TTH_PTH_200_300',
    605: 'TTH_PTH_GT300',
    # bbH
    700: 'BBH_FWDH',
    701: 'BBH',
    # tH
    800: 'TH_FWDH',
    801: 'TH'
}

STXS_STAGE_1_2_DICT_MERGED = {
    0: 'UNKNOWN',
    # Gluon fusion
    100: 'GG2H_FWDH',
    101: 'GG2H_PTH_GT200', # 101, 102, 103, 104 merged into one category
    105: 'GG2H_0J_PTH_0_10',
    106: 'GG2H_0J_PTH_GT10',
    107: 'GG2H_1J_PTH_0_60',
    108: 'GG2H_1J_PTH_60_120',
    109: 'GG2H_1J_PTH_120_200',
    110: 'GG2H_GE2J_MJJ_0_350_PTH_0_60',
    111: 'GG2H_GE2J_MJJ_0_350_PTH_60_120',
    112: 'GG2H_GE2J_MJJ_0_350_PTH_120_200',
    113: 'GG2H_GE2J_MJJ_GT350', # 113, 114, 115, 116 merged into one category
    # VBF
    200: 'QQ2HQQ_FWDH',
    202: 'QQ2HQQ_rest', # 201,202,203,205 merged into one category
    204: 'QQ2HQQ_GE2J_MJJ_60_120', 
    206: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',
    207: 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    208: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25', # 208, 210 merged into one category
    209: 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    # qq -> WH
    300: 'QQ2HLNU_FWDH',
    301: 'VH_lep_PTV_0_150', # 301, 302, 401, 402, 501, 502 merged into one category
    303: 'VH_lep_PTV_GT150', # 303, 304, 305, 403, 404, 405, 503, 504, 505 merged into one category
    # qq -> ZH
    400: 'QQ2HLL_FWDH',
    # gg -> ZH
    500: 'GG2HLL_FWDH',
    # ttH
    600: 'TTH_FWDH',
    601: 'TTH', # 601, 602, 603, 604, 605 merged into one category
}

# STXS Stage 1.2 with GG2H_GE2J_MJJ_GT350 and QQ2HQQ_rest unmerged, rest merged
STXS_STAGE_1_2_DICT_PARTIAL_MERGED = {
    0: 'UNKNOWN',
    # Gluon fusion
    100: 'GG2H_FWDH',
    101: 'GG2H_PTH_GT200', # 101, 102, 103, 104 merged into one category
    105: 'GG2H_0J_PTH_0_10',
    106: 'GG2H_0J_PTH_GT10',
    107: 'GG2H_1J_PTH_0_60',
    108: 'GG2H_1J_PTH_60_120',
    109: 'GG2H_1J_PTH_120_200',
    110: 'GG2H_GE2J_MJJ_0_350_PTH_0_60',
    111: 'GG2H_GE2J_MJJ_0_350_PTH_60_120',
    112: 'GG2H_GE2J_MJJ_0_350_PTH_120_200',
    113: 'GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    114: 'GG2H_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
    115: 'GG2H_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    116: 'GG2H_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',
    # VBF
    200: 'QQ2HQQ_FWDH',
    201: 'QQ2HQQ_0J',
    202: 'QQ2HQQ_1J',
    203: 'QQ2HQQ_GE2J_MJJ_0_60',
    204: 'QQ2HQQ_GE2J_MJJ_60_120', 
    205: 'QQ2HQQ_GE2J_MJJ_120_350',
    206: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',
    207: 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    208: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25', # 208, 210 merged into one category
    209: 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    # qq -> WH
    300: 'QQ2HLNU_FWDH',
    301: 'VH_lep_PTV_0_150', # 301, 302, 401, 402, 501, 502 merged into one category
    303: 'VH_lep_PTV_GT150', # 303, 304, 305, 403, 404, 405, 503, 504, 505 merged into one category
    # qq -> ZH
    400: 'QQ2HLL_FWDH',
    # gg -> ZH
    500: 'GG2HLL_FWDH',
    # ttH
    600: 'TTH_FWDH',
    601: 'TTH', # 601, 602, 603, 604, 605 merged into one category
}

# Mapping for merging STXS 1.2 categories into coarser groups for training
STXS_1_2_MERGE_Helper = {
    101: [101, 102, 103, 104],
    113: [113, 114, 115, 116],
    202: [201, 202, 203, 205],
    208: [208, 210],
    301: [301, 302, 401, 402, 501, 502],
    303: [303, 304, 305, 403, 404, 405, 503, 504, 505],
    601: [601, 602, 603, 604, 605]
}

# Mapping for merging STXS 1.2 partial merged categories (keeps GG2H_GE2J_MJJ_GT350 and QQ2HQQ_rest unmerged)
STXS_1_2_MERGE_Helper_PARTIAL = {
    101: [101, 102, 103, 104],
    208: [208, 210],
    301: [301, 302, 401, 402, 501, 502],
    303: [303, 304, 305, 403, 404, 405, 503, 504, 505],
    601: [601, 602, 603, 604, 605]
}
# Categories to exclude from training
FWDH_CATEGORIES_TO_EXCLUDE = [10, 20, 30, 40, 100, 200, 300, 400, 500, 600]

# Minimum events per category
MIN_EVENTS_PER_CATEGORY = 10

# Mapping between STXS 1.2 merged categories and Run2 categorization
STXS_TO_RUN2_MAPPING = {
    # Gluon fusion (Untagged) categories
    'GG2H_0J_PTH_0_10': ['Untagged_0j_Pt0To10'],
    'GG2H_0J_PTH_GT10': ['Untagged_0j_Pt10To200'],
    'GG2H_1J_PTH_0_60': ['Untagged_1j_Pt0To60'],
    'GG2H_1J_PTH_60_120': ['Untagged_1j_Pt60To120'],
    'GG2H_1J_PTH_120_200': ['Untagged_1j_Pt120To200'],
    'GG2H_GE2J_MJJ_0_350_PTH_0_60': ['Untagged_2j_Pt0To60'],
    'GG2H_GE2J_MJJ_0_350_PTH_60_120': ['Untagged_2j_Pt60To120'],
    'GG2H_GE2J_MJJ_0_350_PTH_120_200': ['Untagged_2j_Pt120To200'],
    'GG2H_GE2J_MJJ_GT350': ['Untagged_2j_mjj350above'],
    'GG2H_PTH_GT200': ['Untagged_Pt200above'],
    # VBF categories
    'QQ2HQQ_GE2J_MJJ_60_120': ['VH_hadronic_tagged_mjj60To120'],
    'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200': ['VBF_2jet_tagged_Pt200above'],
    'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25': ['VBF_2jet_tagged_mjj350To700'],
    'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25': ['VBF_3jet_tagged_mjj350above'],
    'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25': ['VBF_2jet_tagged_mjj700above'],
    'QQ2HQQ_rest': ['VBF_rest', 'VBF_1jet_tagged', 'VH_hadronic_tagged_rest'],
    # VH leptonic categories
    'VH_lep_PTV_0_150': ['VH_leptonic_tagged_Pt0To150'],
    'VH_lep_PTV_GT150': ['VH_leptonic_tagged_Pt150above'],
    # ttH categories
    'TTH': ['ttH_hadronic_tagged', 'ttH_leptonic_tagged'],
}
