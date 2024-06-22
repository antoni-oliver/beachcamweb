// Image uploader js
const imageQuery = "[name^=image]";
const canvasQuery = ".canvas-img";
const formQuery = "form.image-uploader";
const loaderQuery = "div.loader";
const analyze_image_url = "/analyze-image";
const p_show_result = "p-show-result";

/**
 * Initialize ImageUploaderForms events
 */
export function initImageUploaderForms() {
    hideAllSubmits();
    initSubmitEvent();
    initCanvasDrawEvent();
}

/** 
 * Hide all submit buttons initially
 */
function hideAllSubmits() {
    $(`${formQuery} :submit`).hide();
}

/** 
 * Initialize Submit Event delegation
 */
function initSubmitEvent() {
    // Event delegation for form submit
    $(document).on('click', `${formQuery} :submit`, async function (e) {
        e.preventDefault();
        $(this).prop('disabled', true);
        const form = $(this).closest(formQuery);
        showLoader(form);
        let data = await submitForm(form[0]);
        debugger;
        hideLoader(form);
        $(this).prop('disabled', false);
        if (!data) {
            return;
        }
        showResult(form, data);
    });
}

/* 
* Show loader while processing image
*/
function showLoader($form) {
    $form.hide();
    const loader = $(`
        <div class="loader my-5 animate__animated animate__fadeInUp animate__faster">
            <p class="display-6 font-weight-bolder my-5 text-center">Analitzant la teva imatge...</p>
            <div class="spinner-border large text-primary" role="status">
            </div>
        </div>`);
    let form_container = $form.parent();
    form_container.append(loader);
}


/**
 * Submit the form via AJAX
 */
async function submitForm(form) {
    const formData = new FormData(form);
    return fetch(analyze_image_url, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': $('input[name="csrfmiddlewaretoken"]').val()
        }
    })
        .then(response => response.json())
        .then(data => {
            showSuccessMessage();
            return data;
        })
        .catch(() => {
            showErrorMessage();
            return null;
        });

}

function showSuccessMessage() {
    Swal.fire({
        icon: "success",
        title: "Imatge analitzada!",
        showConfirmButton: false,
        timer: 1500
    });
}

function showErrorMessage() {
    Swal.fire({
        icon: "error",
        title: "Error en l'anàlisi de l'imatge",
        showConfirmButton: false,
        timer: 1500
    });
}

/**
 * Hide loader
 */
function hideLoader(form) {
    $(loaderQuery).remove();
    form.show();
}

/* 
* Shows results from predictionDTO
*/
function showResult(form, predictionDTO) {
    var image = new Image();
    image.src = `data:image/png;base64,${predictionDTO.img_predict_content}`;
    const canvas = form.find(canvasQuery).first();
    let submit = form.find(':submit');
    $(`#${p_show_result}`).remove();
    submit.after(`<p id="${p_show_result}" class="h5 ms-3">S'han trobat ${Math.round(predictionDTO.crowd_count)} persones.</p>`);
    drawImageOnCanvas(image, canvas[0]);
}

/**
 * Initialize Canvas Image drawing event delegation
 */
function initCanvasDrawEvent() {
    // Event delegation for canvas viewer
    $(document).on('change', `${formQuery} ${imageQuery}`, function (e) {
        // Check files and if canvas exists
        const form = $(e.target).closest(formQuery);
        const canvas = form.find(canvasQuery).first();
        const submit = form.find(':submit');
        if (!e.target.files.length || !canvas.length) {
            submit.hide();
            return;
        }
        submit.show();
        $(`#${p_show_result}`).remove();
        handleCanvasViewer(e.target.files[0], canvas[0]);
    });
}

/**
 * Display selected form image in canvas
 */
function handleCanvasViewer(image, canvas) {
    try {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.src = e.target.result;
            drawImageOnCanvas(img, canvas)
        };
        reader.readAsDataURL(image);
    } catch (error) {
        console.error(error);
    }
}

/**
 * Display img on canvas
 */
function drawImageOnCanvas(img, canvas) {
    const ctx = canvas.getContext("2d");
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        canvas.style.display = "block";
    };
}
