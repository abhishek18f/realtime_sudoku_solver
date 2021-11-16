import React, { useState } from "react";
import Grid from "./grid.jsx";
import "./index.css";

const Square = ({value, row, col, onCellValueChange}) => (
    <input
        type="text"
        value={value === 0 ? "" : value}
        maxLength="1"
        onChange={(evt) => {
            const cellValue = evt.target.value;
            if (parseInt(cellValue, 10) || cellValue === "") {
                onCellValueChange(row, col, cellValue);
            }
        }}
    />
);

const SudukoBoard = ({ puzzleGrid, onCellValueChange }) => (
    <table className="sudoku">
        <tbody>
        { puzzleGrid.rows.map((row, idx) => (
            <tr key={idx}>
                { row.map(cell => (
                    <td key={cell.col}>
                        <Square
                            value={cell.value}
                            row={cell.row}
                            col={cell.col}
                            onCellValueChange={onCellValueChange}
                        />
                    </td>
                )) }
            </tr>
        )) }
        </tbody>
    </table>
);

export default function Sudoku({ puzzleString }) {
    const [puzzle, setPuzzle] = useState(new Grid(puzzleString));

    function onCellValueEdited (row, col, value) {
        const newGrid = new Grid(puzzle.toFlatString());
        newGrid.rows[row][col].value = value;
        setPuzzle(newGrid);
    }

    return (
        <div className="game">
            {/* <h1>Sudoku Solver</h1> */}
            <SudukoBoard
                puzzleGrid={puzzle}
                onCellValueChange={onCellValueEdited}
            />
        </div>
    );
}
