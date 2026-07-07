// In the ExperimentDataset inline, show each augmentation's fraction input
// only while its checkbox is ticked: id_datasets-<n>-aug_hflip toggles
// id_datasets-<n>-aug_hflip_fraction (same for aug_scale_crop). Progressive
// enhancement only — with JS off the inputs stay visible and the model's
// clean() still validates them.
//
// Vanilla JS on purpose, matching experiment_model_form.js: django.jQuery may
// not be defined yet when this runs.
(function () {
    "use strict";

    var TOGGLES = ["aug_hflip", "aug_scale_crop"];

    function syncCheckbox(checkbox) {
        var fraction = document.getElementById(checkbox.id + "_fraction");
        if (!fraction) {
            return;
        }
        fraction.style.display = checkbox.checked ? "" : "none";
    }

    function syncAll(root) {
        TOGGLES.forEach(function (toggle) {
            root.querySelectorAll('input[type="checkbox"][id$="-' + toggle + '"]')
                .forEach(syncCheckbox);
        });
    }

    function isToggle(el) {
        return el && el.matches && TOGGLES.some(function (toggle) {
            return el.matches('input[type="checkbox"][id$="-' + toggle + '"]');
        });
    }

    function init() {
        syncAll(document);

        document.addEventListener("change", function (event) {
            if (isToggle(event.target)) {
                syncCheckbox(event.target);
            }
        });

        // Django 4.1+ dispatches a native CustomEvent on the freshly added row
        // (it bubbles to document) after "Add another".
        document.addEventListener("formset:added", function (event) {
            if (event.target && event.target.querySelectorAll) {
                syncAll(event.target);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
