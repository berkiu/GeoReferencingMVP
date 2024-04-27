import os

import pandas as pd
import cv2

from shapely.geometry import Polygon
from geopy.distance import great_circle

import folium as fl
import streamlit as st
from streamlit_folium import st_folium



polygon = [
        [-122.4627134068209,37.80742061986032],
        [-122.4624796869529,37.75410839987458],
        [-122.3779746140394,37.754435295918555],
        [-122.38506739030568,37.8165892440382],
        [-122.4627134068209,37.80742061986032]
    ]

p = Polygon(polygon)
center = list(p.centroid.coords)[0]


def get_pos(lat, lng):
    return lat, lng


def get_gps(img_path):
    data_mvp = pd.read_csv('mvp.csv')
    gt, pred = data_mvp[data_mvp['name'] == img_path][['gt', 'pred']].values[0]
    lat_gt, lon_gt = float(gt.split(', ')[0][1:]), float(gt.split(', ')[1][:-1])
    lat_pred, lon_pred = float(pred.split(', ')[0][1:]), float(pred.split(', ')[1][:-1])

    return lat_gt, lon_gt, lat_pred, lon_pred


def initialize_map():
    m = fl.Map((center[1], center[0]), min_zoom=13, max_zoom=17)
    fl.CircleMarker([polygon[0][1], polygon[0][0]], radius=2, tooltip="Upper Left Corner").add_to(m)
    fl.CircleMarker([polygon[1][1], polygon[1][0]], radius=2, tooltip="Lower Left Corner").add_to(m)
    fl.CircleMarker([polygon[2][1], polygon[2][0]], radius=2, tooltip="Lower Right Corner").add_to(m)
    fl.CircleMarker([polygon[3][1], polygon[3][0]], radius=2, tooltip="Upper Right Corner").add_to(m)
    fl.PolyLine([[polygon[0][1], polygon[0][0]], [polygon[1][1], polygon[1][0]]]).add_to(m)
    fl.PolyLine([[polygon[1][1], polygon[1][0]], [polygon[2][1], polygon[2][0]]]).add_to(m)
    fl.PolyLine([[polygon[2][1], polygon[2][0]], [polygon[3][1], polygon[3][0]]]).add_to(m)
    fl.PolyLine([[polygon[3][1], polygon[3][0]], [polygon[4][1], polygon[4][0]]]).add_to(m)

    if st.session_state['user_point']:
        fl.Marker(
            location=[st.session_state['user_point'][0], st.session_state['user_point'][1]],
            tooltip="User GPS",
            icon=fl.Icon(color="red"),
        ).add_to(m)

        fl.PolyLine([[st.session_state['user_point'][0], st.session_state['user_point'][1]], [st.session_state['gt_point'][0], st.session_state['gt_point'][1]]],
                    color='red',
                    dash_array='10').add_to(m)

    if st.session_state['gt_point']:
        fl.Marker(
            location=[st.session_state['gt_point'][0], st.session_state['gt_point'][1]],
            tooltip="Ground Truth",
            icon=fl.Icon(color="green"),
        ).add_to(m)

    if st.session_state['model_point']:
        fl.Marker(
            location=[st.session_state['model_point'][0], st.session_state['model_point'][1]],
            tooltip="Model GPS",
            icon=fl.Icon(color="black"),
        ).add_to(m)

        fl.PolyLine([[st.session_state['model_point'][0], st.session_state['model_point'][1]], [st.session_state['gt_point'][0], st.session_state['gt_point'][1]]],
                    color='black',
                    dash_array='10').add_to(m)

    return m


def initialize_session_state():
    if 'counter' not in st.session_state: 
        st.session_state.counter = 0
    if 'user_point' not in st.session_state: 
        st.session_state.user_point = None
    if 'gt_point' not in st.session_state: 
        st.session_state.gt_point = None
    if 'model_point' not in st.session_state:
        st.session_state.model_point = None


def main():
    initialize_session_state()

    images_path = os.listdir('test_images/')

    img = cv2.imread(os.path.join('test_images/', images_path[st.session_state.counter]))[...,::-1]
    st.image(img)

    m = initialize_map()
    map = st_folium(m, height=450, width=img.shape[1])

    if map.get("last_clicked"):
        lat_gt, lon_gt, lat_pred, lon_pred = get_gps(images_path[st.session_state.counter])

        st.session_state['user_point'] = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])
        st.session_state['gt_point'] = (lat_gt, lon_gt)
        st.session_state['model_point'] = (lat_pred, lon_pred)

        st.rerun()

    if st.session_state['user_point'] is not None:
        model_error = great_circle((st.session_state['gt_point'][0], st.session_state['gt_point'][1]), (st.session_state['model_point'][0], st.session_state['model_point'][1])).m
        user_error = great_circle((st.session_state['gt_point'][0], st.session_state['gt_point'][1]), (st.session_state['user_point'][0], st.session_state['user_point'][1])).m
        st.write('User guess error: {:.3f} m'.format(user_error).rstrip('0').rstrip('.'))
        st.write('Model guess error: {:.3f} m'.format(model_error).rstrip('0').rstrip('.'))

        button = st.button("Next Example", key='button')
        if button:
            for key in st.session_state.keys():
                if key != 'counter':
                    del st.session_state[key]

            st.session_state.counter += 1
            if st.session_state.counter >= len(images_path):
                st.session_state.counter = 0

            st.rerun()


if __name__ == "__main__":
    main()
