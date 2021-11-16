


def sudoku_solver(sudoku , num_of_rows = 9):
    def isSafe(sudoku , row , col , num , num_of_rows):
        #check row
        for r in range(num_of_rows):
            if sudoku[r][col] == num:
                return False
            
        #check cols
        for c in range(num_of_rows):
            if sudoku[row][c] == num:
                return False

        #check mini-square
        miniRow = row//3
        miniCol = col//3
        for r in range(3):
            for c in range(3):
                if sudoku[miniRow*3 + r][miniCol*3 + c] == num:
                    return False
        
        return True

    def solve_sudoku(sudoku , num_of_rows , idx):
        if idx == num_of_rows**2:
            return True

        row = idx//num_of_rows
        col = idx%num_of_rows

        if sudoku[row][col] != 0:
            return solve_sudoku(sudoku , num_of_rows, idx + 1)

        for num in range(1 , num_of_rows + 1):
            if(isSafe(sudoku , row , col , num , num_of_rows)):
                sudoku[row][col] = num
                if(solve_sudoku(sudoku , num_of_rows , idx + 1)):
                    return True
        
        sudoku[row][col] = 0
        return False

    solve_sudoku(sudoku , num_of_rows , 0)
    return sudoku

   
