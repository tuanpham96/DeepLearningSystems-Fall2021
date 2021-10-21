import pickle
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def obtain_result(output_files, google_translator, dest_lang, extra_cols2drop = []):

    info_sel_keys = ['model_info', 'model_config', 'output_files']
    stat_sel_keys = ['total_time', 'accuracies', 'losses']
    cols2drop = ['config_file', 'results', 'checkpoint'] + extra_cols2drop

    all_results = []
    all_translations = {}
    input_translations = []

    # get data
    for i,fn in enumerate(output_files):
        with open(fn, 'rb') as f:
            og_res = pickle.load(f)
        conf = {k:v for k, v in og_res.items() if k in info_sel_keys}
        res = {}
        for k in info_sel_keys:
            res = dict(**res, **conf[k])

        del res['exp_param']

        for k in stat_sel_keys:
            res[k] = og_res[k]

        res['epoch'] = np.linspace(0, og_res['train_config']['num_epochs'], len(res['losses']))

        all_results.append(pd.DataFrame(res))
        all_translations[res['model_name']] = compare2google(google_translator, og_res['translations'], dest_lang=dest_lang)

    # concat and process
    df = pd.concat(all_results, ignore_index=True)\
            .sort_values(by='model_id', ascending=True)\
            .reset_index()\
            .drop(['index'] + cols2drop, axis=1)
    df.accuracies *= 100 # to percentage
    df.total_time /= 60 # to hour

    # get summary
    max_epoch = df.epoch.max()
    final_df = df.query('epoch == @max_epoch').drop(['epoch'], axis=1)
    base_fdf = final_df.query('exp_name == "baseline"')

    def norm_to_base(col, norm_to_percent = True):
        base_val = float(base_fdf[col])
        name_new_col = col+'[norm' + ('_perc' if norm_to_percent else '') + ']'
        final_df[name_new_col] = final_df[col] - base_val
        if norm_to_percent:
            final_df[name_new_col] *= 100/base_val

    norm_to_base('accuracies', False)
    norm_to_base('losses')
    norm_to_base('total_time')

    return df, final_df, base_fdf, all_translations

def compare2google(google_translator, translations, dest_lang, src_lang='en'):
    input_sentences = translations['input']
    input_to_transformer = translations['prediction']

    input_to_google = [google_translator.translate(x, dest=dest_lang).text for x in input_sentences]
    transformer_to_google = [google_translator.translate(x, src=dest_lang, dest=src_lang).text for x in input_to_transformer]

    result_text = ''
    for i in range(len(input_sentences)):
        inp = input_sentences[i]
        inp2transf = input_to_transformer[i]
        inp2google = input_to_google[i]
        transf2google = transformer_to_google[i]

        result_text += '(%d) INPUT: \t\t%s\n' %(i, inp)
        result_text += '    INPUT  --> TRANSF: \t%s\n' %(inp2transf)
        result_text += '    INPUT  --> GOOGLE: \t%s\n' %(inp2google)
        result_text += '    TRANSF --> GOOGLE: \t%s\n' %(transf2google)
        result_text += '-'*200 + '\n'

    return dict(
        input_sentences = input_sentences,
        input_to_transformer = input_to_transformer,
        input_to_google = input_to_google,
        transformer_to_google = transformer_to_google,
        result_text = result_text
    )


def plot_train_progress(df, final_df, base_fdf, main_title,
                        cmap = px.colors.qualitative.Set1):
    max_epoch = df.epoch.max()
    exp_names = df.exp_name.unique().tolist()

    general_layout = general_plotly_layout()

    fig = make_subplots(
        specs=[
            [{'secondary_y':True}],
        ],
    )

    fig.update_layout(general_layout)

    plt_opts_df = []

    for plt_field in ['losses', 'accuracies']:
        splt_args = dict(secondary_y = plt_field == 'accuracies')

        line_style = {} if plt_field == 'accuracies' else dict(line_dash='dot')
        for exp_name in exp_names:
            df_sel = df.query('exp_name == @exp_name')\
                        .sort_values(by='epoch', ascending=True)

            param_name = df_sel.param_name.unique()[0]
            param_vals = df_sel.param_val.unique()

            for i,v in enumerate(param_vals):
                color_id = 0 if exp_name == 'baseline' else i+1
                df_sub = df_sel if v is None else df_sel.query('param_val == @v')
                x_vec = df_sub['epoch']
                y_vec = df_sub[plt_field]

                fig.add_trace(go.Scatter(
                    x = x_vec,
                    y = y_vec,
                    legendgroup = '{}-{}'.format(param_name,v),
                    visible = exp_name == 'baseline',
                    line_color = cmap[color_id],
                    opacity=0.8, line_width=3,
                    **line_style,
                    name = '{}-{}'.format(plt_field,exp_name)
                    ),
                    **splt_args
                )

                plt_opts_df.append(dict(
                    exp_name = exp_name,
                    plt_field = plt_field,
                    param_name = param_name,
                    param_val = v
                ))


    plt_opts_df = pd.DataFrame(plt_opts_df)

    def get_button_dict(exp_name):
        exp_names = plt_opts_df.exp_name
        is_baseline = exp_names == 'baseline'
        plt_fields = plt_opts_df.plt_field
        param_name = plt_opts_df.query('exp_name == @exp_name').param_name.tolist()[0]

        param_vals = np.array(plt_opts_df.param_val)

        if param_name == 'vocab_size_factor':
            param_vals[is_baseline] = 1
        else:
            param_vals[is_baseline] = base_fdf[param_name].tolist()[0]

        # only baseline or selected exp visible
        trace_visible = np.logical_or(is_baseline, exp_names == exp_name)

        # define names, i.e. legends
        trace_names = [
            '{}: {}={} {}'.format(
                f, param_name, v, '(base)' if i else ''
            )
            for f,v,i in zip(plt_fields, param_vals, is_baseline)
        ]


        return dict(
            label = exp_name,
            method = 'update',
            args = [dict(
                visible     = trace_visible,
                name        = trace_names
            )]
        )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="down",
            x=-0.12, y=1, showactive=True, active=-1,
            buttons=[get_button_dict(x) for x in exp_names if x != 'baseline']
        )]
    )

    fig.update_xaxes(title='epoch')
    fig.update_yaxes(title='loss')
    fig.update_yaxes(title='accuracy (%)', secondary_y=True)

    fig.update_layout(
        width  = 1200,
        height = 500,
        legend_tracegroupgap = 25,
        title_text=main_title
    )

    fig.show()


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
        title_x = 0.35,
        title_y = 0.9,
        title_xanchor = 'center',
        title_yanchor = 'top',
        title_font_size=20
    )

    legend_config = dict(
        font = dict(size = 12)
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
