
import numpy as np
import common
import index_definition as idef
import confidence_index as ci
import pandas as pd
def generate_live_match_data(gameids, updates_per_game=10):
    """

    """
    data = pd.read_csv("Linhac24-25_Sportlogiq.csv")
    # gameids = data['gameid'].unique()
    data['inopponentarea'] = data.apply(common.puck_location, axis=1)
    detail_df = ci.plot_confidence_index(data, gameids, idef.CONFIDENCE_INDEX, 30, 0, 3600,Display=False)
    return detail_df



def train():
    pass





