import argparse
import sudoku_solver
#from sudoku_solver import handle_image, sobel_convolution, find_corners, evaluate_corners, segment_cells

def handle_arguments():
    print("handling arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", default = "sudoku_far_away.jpg", type=str, help="File path to image of sudoku")

    args = parser.parse_args()
    infile = args.input_image

    print(infile)

    sudokuImage = sudoku_solver.SudokuImage(infile)

    image_array = sudokuImage.handle_image()

    #image_array = handle_image(infile)

    gradient_array = sudokuImage.sobel_convolution(image_array)
 
    corner_position_list = sudokuImage.find_corners(gradient_array)
    corner_position_list = sudokuImage.evaluate_corners(gradient_array, corner_position_list)
    normalized_image = sudokuImage.normalize_image(infile, corner_position_list, 500)
    #gradient_array = sudoku_solver.sobel_convolution(normalized_image)
    solution_matrix = sudokuImage.segment_cells(normalized_image)
    #sudoku_solver.print_solution_matrix(solution_matrix)
    #sudoku_solver.solve_sudoku(solution_matrix)

    mySudokuMatrix = sudoku_solver.SudokuMatrix(solution_matrix)
    mySudokuMatrix.print_solution_matrix()
    #mySudokuMatrix.print_working_matrix()
    mySudokuMatrix.solve_sudoku()
    


if __name__=='__main__':
    handle_arguments()