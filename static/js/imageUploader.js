// Image uploader js

const imageQuery = "[name^=image]";
const canvasQuery = ".canvas-img";
const formQuery = "form.image-uploader";

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
        let response = await submitForm(form);
        hideLoader(form);
        showFeedback(response);
        $(this).prop('disabled', false);
    });
}

/**
 * Submit the form via AJAX
 */
function submitForm(form) {
    return fetch()
}

/* 
* Show loader while processing image
*/
function showLoader($form) {
    $form.hide();
    const loader = $(`
        <div class="loader my-5 animate__animated animate__fadeInUp animate__faster">
            <p class="display-6 font-weight-bolder my-5 text-center">Analitzant sa teva imatge...</p>
            <div class="spinner-border large text-primary" role="status">
            </div>
        </div>`);
    let form_container = $form.parent();
    form_container.append(loader);
}

function hideLoader(form) {

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
        handleCanvasViewer(e.target.files[0], canvas[0]);
    });
}

/**
 * Display the selected image on the canvas
 */
function handleCanvasViewer(image, canvas) {
    try {
        const reader = new FileReader();
        reader.onload = (e) => {
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.src = e.target.result;
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                canvas.style.display = "block";
            };
        };
        reader.readAsDataURL(image);
    } catch (error) {
        console.error(error);
    }
}
