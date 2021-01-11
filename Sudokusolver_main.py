import argparse
import sudoku_solver
#from sudoku_solver import handle_image, sobel_convolution, find_corners, evaluate_corners, segment_cells

def handle_arguments():
    print("handling arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", default = "sudoku_extrabild.jpg", type=str, help="File path to image of sudoku")

    args = parser.parse_args()
    infile = args.input_image

    print(infile)

    sudokuImage = sudoku_solver.SudokuImage(infile)

    image_array = sudokuImage.handle_image()

    #image_array = handle_image(infile)

    gradient_array = sudoku_solver.sobel_convolution(image_array)
 
    corner_position_list = sudoku_solver.find_corners(gradient_array)
    corner_position_list = sudoku_solver.evaluate_corners(corner_position_list)
    normalized_image = sudokuImage.normalize_image(infile, corner_position_list, 500)
    #gradient_array = sudoku_solver.sobel_convolution(normalized_image)
    sudoku_solver.segment_cells2(normalized_image, 100)
    #segment_cells(result_array, result_array, corner_position_list)
    

if __name__=='__main__':
    handle_arguments()