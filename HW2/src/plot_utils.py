import time, os, glob
import pandas as pd
import numpy as np
import yaml

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def load_data(exp_file_pref):
    stat_file = exp_file_pref + '_exp_stat.csv'
    conf_file = exp_file_pref + '_exp_conf.yml'
    conf = yaml.load(open(conf_file, 'r'))
    df = pd.read_csv(stat_file)

    params = conf['param']
    param_keys = list(params.keys())
    return df, conf, param_keys

def construct_expvar(df, param_keys, return_df=False):
    df['exp_var'] = df[param_keys].astype(str).agg(', '.join, axis=1)
    if return_df:
        return df

def get_test_and_totaltime(df):
    max_epoch = df['epoch'].max()

    total_time_df = df.filter(regex=r'(exp_var|_time$)', axis=1)\
                        .groupby('exp_var').sum().sum(axis=1)\
                        .to_frame(name='total_time').reset_index()
    total_time_df['total_time'] /= 60

    test_acc_df = df.query('epoch == @max_epoch')[['exp_var','test_acc']]

    test_and_time_df = pd.merge(test_acc_df, total_time_df, on = ['exp_var'])
    sorted_expvar_bytest = test_and_time_df.sort_values('test_acc', ascending=False)['exp_var']
    return test_and_time_df, sorted_expvar_bytest


def general_plotly_layout():
    axis_config = dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(0, 0, 0)',
        linewidth=2,
        ticks='outside',
        tickwidth=2,
    )

    font_config = dict(
        family="Fira Sans",
        size=15,
        color='black'
    )

    title_config = dict(
        title_x = 0.5,
        title_y = 0.98,
        title_xanchor = 'center',
        title_yanchor = 'top',
        title_font_size=25
    )

    legend_config = dict(
        font = dict(size = 15)
    )

    general_layout = go.Layout(
        xaxis=axis_config,
        yaxis=axis_config,
        font=font_config,
        margin=dict(
            autoexpand=True,
            l=100,
            r=50,
            t=100,
            b=120
        ),
        plot_bgcolor='white',
        autosize=True,
        legend = legend_config,
        **title_config
    )

    for i in range(1,8):
        general_layout['xaxis' + str(i)] = axis_config
        general_layout['yaxis' + str(i)] = axis_config
    return general_layout

def set_dict_keyval_default(d, k, v):
    if k not in d:
        d[k] = v
def set_dict_default(d, d0):
    for k, v in d0.items():
        set_dict_keyval_default(d, k, v)

def plot_benchmark(df, main_title, variation_label,
                   colors = px.colors.qualitative.Plotly,
                   sort_bar_by_test = True,
                   subplot_args = dict(), layout_args = dict()):

    test_and_time_df, sorted_expvar_bytest = get_test_and_totaltime(df)
    unq_var = df['exp_var'].unique()
    max_epoch = df['epoch'].max()

    general_layout = general_plotly_layout()

    phases = ['train', 'valid']

    phase_opt = dict(
        train = dict(line_dash='dot'),
        valid = dict()
    )

    metrics = ['acc', 'loss']
    progress_labels = dict(
        acc = 'accuracy %',
        loss = 'loss',
    )

    set_dict_default(subplot_args, dict(
        horizontal_spacing = 0.15,
        vertical_spacing = 0.2))


    fig = make_subplots(
        rows=2, cols=5,
        specs=[
            [{'colspan':3}, None, None, {'colspan':2, 'secondary_y':True}, None],
            [{'colspan':3}, None, None, {'colspan':2}, None]
        ],
        subplot_titles=['train progress', 'test & total time',
                        'loss progress', 'train/validation time per epoch'],
        **subplot_args

    )


    fig.update_layout(general_layout)

    phasemtric_dict = dict(
        phase = [],
        metric = []
    )

    always_visible = []

    test_spltpos = dict(row = 1, col = 4)
    time_spltpos = dict(row = 2, col = 4)

    for i_met, metric in enumerate(metrics):
        splt_pos = dict(
            row = i_met+1,
            col = 1
        )

        for i_var, var in enumerate(unq_var):
            df_sel = df.query('exp_var == @var')
            x_vec = df_sel['epoch']

            for i_phase, phase in enumerate(phases):
                y_vec = df_sel[phase + '_' + metric]

                phasemtric_dict['phase'].append(phase)
                phasemtric_dict['metric'].append(metric)

                fig.add_trace(go.Scatter(
                    x = x_vec,
                    y = y_vec,
                    name = '%s (%s)' %(phase, var),
                    legendgroup = var,
                    opacity=0.9,
                    showlegend = i_met == 0,
                    line_color = colors[i_var],
                    **phase_opt[phase]
                    ),
                    **splt_pos
                )

        fig.update_yaxes(title = progress_labels[metric], **splt_pos)

        if i_met == len(metrics)-1:
            fig.update_xaxes(title = 'epoch', **splt_pos)

    # test acc and total time
    fig.add_trace(go.Bar(
        x = test_and_time_df['exp_var'],
        y = test_and_time_df['test_acc'],
        opacity=0.9,
        showlegend = False,
        marker_color = colors,
        width = 0.8,
        ),
        **test_spltpos,
        secondary_y=False,
    )

    always_visible.append(True)
    fig.update_xaxes(title = variation_label, **test_spltpos)
    if sort_bar_by_test:
        fig.update_xaxes(
            categoryorder = 'array',
            categoryarray = sorted_expvar_bytest,
            **test_spltpos)
    fig.update_yaxes(title = 'test accuracy (bar)', **test_spltpos, secondary_y=False)

    fig.add_trace(go.Scatter(
        x = test_and_time_df['exp_var'],
        y = test_and_time_df['total_time'],
        showlegend = False,
        mode = 'markers',
        marker = dict(
            color = 'rgba(250,250,250,0.4)',
            size = 20,
            symbol = 'star-diamond',
            line_width = 3
        )
        ),
        **test_spltpos,
        secondary_y=True,
    )
    always_visible.append(True)
    fig.update_yaxes(title = 'total time (minutes, diamond)', **test_spltpos, secondary_y=True)

    # time
    for i_var, var in enumerate(unq_var):
        df_sel = df.query('exp_var == @var')

        x_vec = df_sel['train_time']
        y_vec = df_sel['valid_time']

        common_pltsty = dict(
            mode = 'markers',
            line_color = colors[i_var],
            showlegend = False,
            legendgroup = var,
            name = var
        )

        fig.add_trace(go.Scatter(
            x = x_vec,
            y = y_vec,
            opacity=0.25,
            marker_size=10,
            **common_pltsty
            ),
            **time_spltpos
        )
        always_visible.append(True)

        fig.add_trace(go.Scatter(
            x = [x_vec.mean()],
            y = [y_vec.mean()],
            opacity=0.8,
            marker_size=15,
            **common_pltsty
            ),
            **time_spltpos
        )
        always_visible.append(True)

    fig.update_xaxes(title = 'train time (sec)', **time_spltpos)
    fig.update_yaxes(title = 'validation time (sec)', **time_spltpos)

    phasemtric_dict = pd.DataFrame(phasemtric_dict)

    def get_visible_vec(phase='both'):
        if phase == 'both':
            vis_vec = [True]*len(phasemtric_dict.phase)
        else:
            vis_vec = list(phasemtric_dict.phase == phase)
        return vis_vec + always_visible

    def get_button_dict(label='Both', phase='both', method='update'):
        return dict(
            label = label,
            method = method,
            args = [{"visible": get_visible_vec(phase)}]
        )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.16,
            y=1.1,
            showactive=True,
            buttons=list([
                get_button_dict('Both', 'both'),
                get_button_dict('Train', 'train'),
                get_button_dict('Validate', 'valid')
            ]),
        )]
    )

    for i in fig['layout']['annotations']:
        i['font']['size']=20

    set_dict_default(layout_args, dict(
        width  = 1500,
        height = 800,
        legend_tracegroupgap = 25,
    ))

    fig.update_layout(
        title_text=main_title,
        **layout_args
    )


    fig.show()
