import React, { useEffect, useState } from "react";



function UploadForm() {

    const [file, setFile] = useState(null);
    const [error, setError] = useState(null);
    
    const allowedTypes = ['image/png' , 'immage/jpg', 'image/jpeg']
    const changeHandler = (e) => {
        let selected = e.target.files[0];
        
        if(selected && allowedTypes.includes(selected.type)){
            console.log(selected);
            setFile(selected);
            setError('');
        }
        else{
            setFile(null);
            setError("Please select an image file.")
        }
    }

    return (
        <div>
            <form>
                <label>
                    <input type="file" onChange = {changeHandler} />
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
