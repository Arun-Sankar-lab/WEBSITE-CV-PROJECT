
document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const imageUpload = document.getElementById('image-upload').files[0];
    const formData = new FormData();
    formData.append('image', imageUpload);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.image) {
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + data.image;
            img.id = 'uploaded-image';
            img.style.display = 'block';
            img.style.margin = '20px auto';
            document.getElementById('output').innerHTML = ''; 
            document.getElementById('output').appendChild(img);
        } else {
            alert('Error uploading image');
        }
    })
    .catch(error => console.error('Error:', error));
});

document.getElementById('low-threshold').addEventListener('input', function () {
    document.getElementById('low-threshold-value').textContent = this.value;
});

document.getElementById('high-threshold').addEventListener('input', function () {
    document.getElementById('high-threshold-value').textContent = this.value;
});


document.getElementById('kernel-size').addEventListener('input', function () {
    document.getElementById('kernel-size-value').textContent = this.value;
});

document.getElementById('apply-canny').addEventListener('click', function () {
    const lowThreshold = document.getElementById('low-threshold').value;
    const highThreshold = document.getElementById('high-threshold').value;
    const uploadedImage = document.getElementById('uploaded-image');

    if (!uploadedImage) {
        alert('Please upload an image first.');
        return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(uploadedImage, 0, 0);
    const imageData = canvas.toDataURL('image/png').split(',')[1];  // Rem

    // Send the image and threshold values to the backend for Canny processing
    fetch('/canny', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: imageData,
            low_threshold: lowThreshold,
            high_threshold: highThreshold
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.canny_image) {
            // Display the Canny edge detection result
            const cannyImage = document.createElement('img');
            cannyImage.src = 'data:image/png;base64,' + data.canny_image;
            cannyImage.id = 'canny-image';
            cannyImage.style.display = 'block';
            cannyImage.style.margin = '20px auto';
            document.getElementById('output').innerHTML = '';  // Clear previous content
            document.getElementById('output').appendChild(cannyImage);
        } else {
            alert('Error applying Canny edge detection');
        }
    })
    .catch(error => console.error('Error:', error));
});
