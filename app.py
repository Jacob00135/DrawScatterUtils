from flask import Flask, render_template, request
from draw_graph import test_draw_whole_scatter, test_draw_magnify_scatter, test_draw_subgraph_scatter

app = Flask(__name__)


@app.route('/')
def index():
    graph_type = request.args.get('graph_type', 'whole')
    filename_list = eval('test_draw_{}_scatter'.format(graph_type))()
    return render_template('index.html', filename_list=filename_list)


if __name__ == '__main__':
	app.run(debug=True, port=5000)
