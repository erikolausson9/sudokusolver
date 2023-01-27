Imaged-based Sudoku-solver

Intro

The idea of this project is to create an automated Sudoku-solver that, given an image of the Sudoku in the local newspaper, can solve it within a minute.

Two main challanges are inherit in this:
1) In the input-image, find the Sudoku-board, segement out the 81 squares that are either empty or filled with a number, and interpret the numbers that are there correctly.
2) Given this starting-position, iterate over the board and fill in the missing numbers, i.e. solve the actual number-puzzle.
Below I will go through the steps I've taken in greater detail.





