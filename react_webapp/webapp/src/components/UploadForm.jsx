import React, { useEffect, useState } from "react";



function UploadForm() {

    const [file, setFile] = useState(null);
    const [error, setError] = useState(null);
    
    const allowedTypes = ['image/png' , 'immage/jpg', 'image/jpeg']
    const changeHandler = (e) => {
        e.preventDefault()
        let selected = e.target.files[0];
        
        if(selected && allowedTypes.includes(selected.type)){
            console.log(e.target.files[0]);
            setFile(selected);
            setError('');
            
            const formData = new FormData();
            formData.append('name', 'Image Upload');
            formData.append("img" , selected)
            console.log(formData)
            
            const Upload = async() => {
              await fetch('/process', {
                method: 'POST',
                body: formData,
                headers: {
                    'Content-Type': 'multipart/form-data; ',
                  },
              }).then(resp => {
                resp.json().then(data => {console.log(data)})
              })
            }
            Upload();
        }
        else{
            setFile(null);
            setError("Please select an image file.")
        }
    }

    return (
        <div>
            <form method='POST'>
                <label>
                    <input type="file" onChange = {changeHandler} />
                    {/* <img> file </img> */}
                    <span>+</span>
                </label>
                <div className = "output">
                    {error && <div className = "error">{error}</div>}
                    {file && <div>{ file.name }</div>}
                </div>
            </form>
        </div>
    )
}

export default UploadForm
