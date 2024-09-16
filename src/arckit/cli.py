import sys
import argparse
from .data import load_single
from .vis import draw_task, output_drawing


def task_print():

    if len(sys.argv) < 2:
        print("Usage: arctask <task_id>")
        return

    task = load_single(sys.argv[1])
    task.show()


def task_save():
    parser = argparse.ArgumentParser(description='Save a task to a image file.')
    parser.add_argument('--task_id', '-i', type=str, help='The task id to save. Can either be a task ID or a string e.g. `train0`')
    parser.add_argument('--width' ,  '-w', type=int, help='The width of the output image', default=20)
    parser.add_argument('--height',  '-h', type=int, help='The height of the output image', default=10)
    parser.add_argument('--output',  '-o', type=str, help='The output file to save to. Must end in .svg/.pdf/.png. By default, pdf is used.', default=False, required=False)
    args = parser.parse_args()

    task = load_single(args.task_id)
    drawing = draw_task(task, width=args.width, height=args.height)

    out_fn = args.output \
                  or f'{task.id}_{drawing.width} x {drawing.height}.pdf'
    print(f'Drawn task {task.id} ({drawing.width} x {drawing.height}), saving to {out_fn}')
    output_drawing(drawing, out_fn)