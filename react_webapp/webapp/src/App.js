// import logo from './logo.svg';
import './App.css';
// import ImageUpload from "./components/ImageUpload.jsx";
import React, {useState} from 'react';
import Title from './components/Title';
import UploadForm from './components/UploadForm';
import Sudoku from './components/board';

function App() {
  return (
    <div className="App">
      {/* <ImageUpload/> */}
      <Title/>
      <UploadForm/>
      <Sudoku puzzleString="4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......" />,
      
  
    </div>
  );
}

export default App;
