import argparse
from sudoku_solver import handle_image, sobel_convolution, find_corners

def handle_arguments():
    print("handling arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", default = "sudoku_extrabild.jpg", type=str, help="File path to image of sudoku")

    args = parser.parse_args()
    infile = args.input_image

    print(infile)

    image_array = handle_image(infile)

    result_array = sobel_convolution(image_array)
 
    find_corners(result_array)


if __name__=='__main__':
    handle_arguments()