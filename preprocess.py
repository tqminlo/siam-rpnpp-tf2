import numpy as np


def get_anchors(temp_rate_size, anchor_rates):
    anchor_rates = np.array(anchor_rates).transpose()
    temp_rate_size = np.array(temp_rate_size)
    anchors = anchor_rates.dot(temp_rate_size)
    return anchors


def get_y_true(box_rate_data, map_size, anchors):
    '''
    :param
        box_rate_data: shape (batch, data), with data format: (x, y, h, w)
        map_size: 25
        anchors: array of anchor: shape (num_anchors, 2), format: ((h1, w1), (h2, w2), ...)
    :return: (num_anchors = len(anchors))
        cls: shape (batch, map_size, map_size, num_anchors)
        loc: shape (batch, map_size, map_size, 4*num_anchors)
    '''

    num_anchors = len(anchors)
    batch = len(box_rate_data)
    cls = np.zeros(shape=(batch, map_size, map_size, num_anchors), dtype=np.float32)
    loc = np.zeros(shape=(batch, map_size, map_size, 4*num_anchors), dtype=np.float32)

    box_rate_sizes = box_rate_data[:, 2:]    # (batch, 2)
    box_rate_center = box_rate_data[:, :2]   # (batch, 2)
    box_map_center = (box_rate_center * map_size - 0.0001).astype(int)
    anchors_trans = anchors.transpose()    # (2, num_anchors)
    compare_box_anchors = box_rate_sizes * (1/anchors_trans) + (1/box_rate_center) * anchors_trans   # (batch, num_anchors)
    compare_box_anchors = np.argmin(compare_box_anchors, axis=1)   # (batch,)

    for i in range(batch):
        anchor_id = compare_box_anchors[i]
        map_x, map_y = box_map_center[i][0], box_map_center[i][1]
        cls[i][map_x][map_y][anchor_id] = 1

        reg_size = np.log(box_rate_sizes[i] / anchors[anchor_id])
        anchor_center = (box_map_center[i] + 0.5) / 25    # (2,)
        reg_center = box_rate_center[i] - anchor_center    # (2,)
        reg_data = np.array(reg_center[0], reg_center[1], reg_size[0], reg_size[1])
        loc[i][map_x][map_y][4*anchor_id : 4*(anchor_id+1)] = reg_data

    return cls, loc



if __name__ == "__main__":
    a = np.array([2,2,2])
    print(np.log(a))


