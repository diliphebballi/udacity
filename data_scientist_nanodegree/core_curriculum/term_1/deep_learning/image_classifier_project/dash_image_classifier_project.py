# Image classifier dash application
# python dash_image_classifier_project.py   

import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import base64
import image_classifier_project

DEFAULT_TOP_K = 5
DEFAULT_MEDIA_DIRECTORY = 'media'
EXTERNAL_STYLESHEETS = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
    {
        'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u',
        'crossorigin': 'anonymous'
    },
    # 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css',
    # {
    #     'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css',
    #     'rel': 'stylesheet',
    #     'integrity': 'sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp',
    #     'crossorigin': 'anonymous'
    # },
    #'https://codepen.io/chriddyp/pen/bWLwgP.css'
    #'https://codepen.io/chriddyp/pen/bWLwgP.css'
]


def remove_all_file_from_folder(folder_path):
    '''
    Remove all file and subfolder from a folder
    '''
    print('Remove all file from media folder\n\t{}'.format(DEFAULT_MEDIA_DIRECTORY))
    for file_object in os.listdir(folder_path):
        file_object_path = os.path.join(folder_path, file_object)
        if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)


def save_image(name, content):
    '''
    Decode and store a file uploaded with Plotly Dash

    '''
    image_filepath = os.path.join(DEFAULT_MEDIA_DIRECTORY, name)
    data = content.encode('utf8').split(b';base64,')[1]
    with open(image_filepath, 'wb') as fp:
        print('Save uploaded image\n\t{}'.format(image_filepath))
        fp.write(base64.decodebytes(data))


def get_encoded_image(image_filepath):
    '''
    '''
    encoded_image = base64.b64encode(open(image_filepath, 'rb').read())
    file_name, extension = image_filepath.split('.')
    return 'data:image/{};base64,{}'.format(extension.lower(), encoded_image.decode())


def _create_app():
    ''' 
    Creates dash application

    Returns:
        app (dash.Dash): Dash application
    '''

    app = dash.Dash(__name__, external_stylesheets = EXTERNAL_STYLESHEETS)

    print('Check if media folder present and if not create it\n\t{}'.format(DEFAULT_MEDIA_DIRECTORY))
    if not os.path.exists(DEFAULT_MEDIA_DIRECTORY):
        os.makedirs(DEFAULT_MEDIA_DIRECTORY)

    model, category_label_to_name = image_classifier_project.load_classifier()

    app.layout = html.Div(
        [
        dbc.Nav(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.A('Image Classifier Application', href='/', className = 'navbar-brand' )
                            ], className = 'navbar-header')
                        , html.Div(
                            [
                                html.Ul(
                                    [
                                        html.Li(html.A('Made with Udacity', href='https://www.udacity.com/'))
                                        , html.Li(html.A('Github', href='https://github.com/simonerigoni/udacity/tree/master/data_scientist_nanodegree/core_curriculum/term_1/deep_learning/image_classifier_project'))
                                    ], className = 'nav navbar-nav')
                            ], className = 'collapse navbar-collapse')
                    ], className = 'container')
            ], className = 'navbar navbar-inverse navbar-fixed-top')
            , html.Div(
                [
                    html.H1('Image Classifier Application', className = 'text-center')
                    , html.P('Classify Images of Flowers', className = 'text-center')
                    , html.Hr()
                    , html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Upload(id = 'upload-image', children = html.Div(['Drag and Drop or ', html.A('Select File')]),
                                        style = {
                                            'width': '98%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple = True
                                    )
                                    , html.Div(id = 'output-image-upload')
                                    , html.Hr()
                                    , html.Button('Classify Image', id = 'button-submit', className = 'btn btn-lg btn-success')
                                ] , className = 'row')
                        ] , className = 'container')
                ], className = 'jumbotron')
            , html.Div(id = 'results')
    ] , className = 'container')


    def parse_contents(contents, filename, date):
        '''
        Parse loaded image
        '''
        return html.Div([
            #html.H5(filename)
            #, html.H6(datetime.datetime.fromtimestamp(date))
            ## HTML images accept base64 encoded strings in the same format that is supplied by the upload
            # ,
            html.Img(src = contents)
            #, html.Hr()
            #, html.Div('Raw Content')
            #, html.Pre(contents[0:200] + '...', style = {'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'})
        ])


    @app.callback(dash.dependencies.Output('output-image-upload', 'children'), [dash.dependencies.Input('upload-image', 'contents')], [dash.dependencies.State('upload-image', 'filename'), dash.dependencies.State('upload-image', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        '''
        Show loaded image
        '''
        if list_of_contents is not None:
            children = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
            save_image(list_of_names[0], list_of_contents[0])
            return children[0]

    @app.callback(dash.dependencies.Output('results','children'), [dash.dependencies.Input('button-submit', 'n_clicks')])
    def update_results(n_click):
        '''
        Update the results section 

        Args:
            n_click (int): value of n_clicks of button-submit

        Returns:
            results (list): list of dash components 
        '''
        results = []
        files = os.listdir(DEFAULT_MEDIA_DIRECTORY)
        number_of_classes = image_classifier_project.get_number_of_classes()
        sample_training_dataset = image_classifier_project.get_sample_from_training_dataset(category_label_to_name)
        if len(files) == 0:
            results.append(html.Div(
                [
                    html.H2('Overview of Training Dataset', className='text-center')
                    , html.H3('{} classes'.format(number_of_classes), className = 'text-center')
                ]))

            buffer_categories = []
            for category in list(sample_training_dataset.keys()):
                #results.append(html.Div([html.Img(src = get_encoded_image(sample_training_dataset[category]), style = {'height':'20%', 'width':'20%'}), html.H5(category)])) 
                if len(buffer_categories) == 4:
                    #print(buffer_categories[0], buffer_categories[1], buffer_categories[2], buffer_categories[3])
                    results.append(html.Div([
                        html.Div([html.Img(src = get_encoded_image(sample_training_dataset[buffer_categories[0]]), style = {'height':'75%', 'width':'75%'}), html.H5(buffer_categories[0])], className = 'col-sm-3')
                        , html.Div([html.Img(src = get_encoded_image(sample_training_dataset[buffer_categories[1]]), style = {'height':'75%', 'width':'75%'}), html.H5(buffer_categories[1])], className = 'col-sm-3')
                        , html.Div([html.Img(src = get_encoded_image(sample_training_dataset[buffer_categories[2]]), style = {'height':'75%', 'width':'75%'}), html.H5(buffer_categories[2])], className = 'col-sm-3')
                        , html.Div([html.Img(src = get_encoded_image(sample_training_dataset[buffer_categories[3]]), style = {'height':'75%', 'width':'75%'}), html.H5(buffer_categories[3])], className = 'col-sm-3')
                    ], className = 'row'))
                    buffer_categories = []
                buffer_categories.append(category)
        else:
            image_filepath = DEFAULT_MEDIA_DIRECTORY + '/' + files[0]
            print('Classify image\n\t{}'.format(image_filepath))
            category_probability = image_classifier_project.get_prediction(model, category_label_to_name, image_filepath)
            
            print(category_probability)

            results.append(html.Div(
                [
                    html.H2('Classification', className='text-center')
                    , dcc.Graph(
                        figure = go.Figure(
                            data = 
                            [
                                go.Bar(
                                    x = list(category_probability.keys())
                                    , y = list(category_probability.values())
                                    , name = 'Probability'
                                    , marker = go.bar.Marker(color = 'rgb(55, 83, 109)')
                                )
                            ]
                            , layout = go.Layout(
                                title = 'Categories Probability Distribution'
                                , showlegend = False
                                , legend = go.layout.Legend(x = 0, y = 1.0)
                                , margin = go.layout.Margin(l = 40, r = 0, t = 40, b = 30)
                            )
                        )
                        , style = {'height': 600}
                        , id = 'categories-distribution-graph')
                ]))

            remove_all_file_from_folder(DEFAULT_MEDIA_DIRECTORY)
        return results

    return app


if __name__ == '__main__':
    app = _create_app()
    app.run_server(debug = True)