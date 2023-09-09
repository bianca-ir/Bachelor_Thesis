const uploadBtn = document.getElementById('uploadBtn');
const uploadedImage = document.getElementById('uploadedImage');

let filePath; 
let filename; 


const algorithmSelect = document.getElementById('algorithmSelect');
const svmParams = document.getElementById('svmParams');
const rfParams = document.getElementById('rfParams');
const cnnParams = document.getElementById('cnnParams');

algorithmSelect.addEventListener('change', function() {
  
  
  svmParams.style.display = 'none';
  rfParams.style.display = 'none';
  cnnParams.style.display = 'none';

  
  if (algorithmSelect.value === 'svm') {
    svmParams.style.display = 'block';
  
  } else if (algorithmSelect.value === 'rf') {
    rfParams.style.display = 'block';

  } else if (algorithmSelect.value === 'cnn') {
    cnnParams.style.display = 'block';
  
  }
});

uploadBtn.addEventListener('change', function(e) {

  const file = e.target.files[0];

  if (file) {
    const reader = new FileReader();

    reader.addEventListener('load', function() {
      const imageDataURL = reader.result;
      uploadedImage.src = imageDataURL;
      uploadedImage.style.display = 'block';

    
      const formData = new FormData();
      formData.append('file', file);

     
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.addEventListener('load', function() {
        canvas.width = 256;
        canvas.height = 256;
        ctx.drawImage(img, 0, 0, 256, 256);

        const resizedImageDataURL = canvas.toDataURL();
        uploadedImage.src = resizedImageDataURL;
      });

      img.src = imageDataURL;

      
      fetch('/upload', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
        
      
          const sentFile = data.filename;

         
          predictBtn.dataset.filename = sentFile;
        })
        .catch(error => {
          console.error('Upload request error:', error);
        });
    });

    reader.readAsDataURL(file);
  }
});




predictBtn.addEventListener('click', function() {
  const selectedAlgorithm = algorithmSelect.value;
  const selectedConfigurationSVM = customDropdownSVM.value;
  const selectedConfigurationRF = customDropdownRF.value;
  const selectedConfigurationCNN = customDropdownCNN.value;
  const filename = predictBtn.dataset.filename;
  let parameters;

  if (selectedAlgorithm === 'svm') {
   

    parameters = {
      algorithm: selectedAlgorithm,
      configuration: selectedConfigurationSVM,
      filename: filename
    };
  } else if (selectedAlgorithm === 'rf') {
    

    parameters = {
      algorithm: selectedAlgorithm,
      configuration: selectedConfigurationRF,
      filename: filename
    };
  } else if (selectedAlgorithm === 'cnn') {
    

    parameters = {
      algorithm: selectedAlgorithm,
      configuration: selectedConfigurationCNN,
      filename: filename
    };
  } else {
    console.error('Invalid algorithm selected');
    return;
  }

  const predictionAccuracy = document.getElementById('predictionAccuracy');
  predictionAccuracy.textContent = 'Loading...';

 
  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      filename: filename,
      parameters: parameters
    })
  })
    .then(response => response.json())
    .then(data => {
   
      const predictedClass = data.predicted_class;
      const malignantProb = data.malignant_probability; 
      const benignProb = data.benign_probability; 
      predictionAccuracy.textContent = ''; 
      const lines = [
        `Predicted class: ${predictedClass}`,
        `Malignant probability: ${malignantProb}`,
        `Benign probability: ${benignProb}`
      ];

      lines.forEach(line => {
        const paragraph = document.createElement('p');
        paragraph.textContent = line;
        predictionAccuracy.appendChild(paragraph);
      });
    })
    .catch(error => {
      console.error('Prediction request error:', error);
    });
});
