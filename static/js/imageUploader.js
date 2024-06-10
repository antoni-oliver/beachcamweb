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
    $(`${formQuery} :submit`).each(function () {
        $(this).hide();
    });
}

/** 
 * Initialize Submit Event delegation
 */
function initSubmitEvent() {
    // Event delegation for form submit
    $(document).on('click', `${formQuery} :submit`, function (e) {
        e.preventDefault();
        const form = $(this).closest(formQuery)[0];
        submitForm(form);
    });
}

/**
 * Submit the form via AJAX
 */
function submitForm(form) {
    // TODO: implement ajax
    alert("PALLA QUE VOY");
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
        if (e.target.files.length <= 0 || !canvas.length) {
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
