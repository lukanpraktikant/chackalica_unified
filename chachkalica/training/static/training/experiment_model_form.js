// Show only the builder-option fields (rendered by ExperimentModelForm) that
// belong to the architecture currently selected in each ExperimentModel inline
// row. Fields are tagged with class "xm-spec-field" and data-arch="<arch>"; we
// toggle the enclosing admin .form-row so labels/help hide with the widget.
//
// Vanilla JS on purpose: relying on django.jQuery meant the script ran before
// jQuery was defined and threw, leaving every arch's fields visible.
(function () {
    "use strict";

    function syncRow(row) {
        if (!row || !row.querySelectorAll) {
            return;
        }
        var archSelect = row.querySelector('select[id$="-arch"]');
        if (!archSelect) {
            return;
        }
        var arch = archSelect.value;
        row.querySelectorAll(".xm-spec-field").forEach(function (field) {
            var matches = field.getAttribute("data-arch") === arch;
            var formRow = field.closest(".form-row") || field;
            formRow.style.display = matches ? "" : "none";
        });
    }

    function rowFor(el) {
        return el.closest(".inline-related") || el.closest("tr") || el.closest(".form-row") || el;
    }

    // Selecting an RF-DETR variant fills the resolution field with that variant's
    // native square resolution (read from the select's data-native-resolutions map),
    // giving the user the right starting point to edit from.
    function fillRfdetrResolution(variantSelect) {
        var map;
        try {
            map = JSON.parse(variantSelect.getAttribute("data-native-resolutions") || "{}");
        } catch (err) {
            return;
        }
        var native = map[variantSelect.value];
        if (native === undefined) {
            return;  // "(default)" or a custom value — leave the field alone
        }
        var row = rowFor(variantSelect);
        var resInput = row.querySelector('input[id$="-xm_rfdetr_resolution"]');
        if (resInput) {
            resInput.value = native;
        }
    }

    function init() {
        // Skip the hidden empty-form template; real rows get cloned from it.
        document.querySelectorAll(".inline-related:not(.empty-form)").forEach(syncRow);

        document.addEventListener("change", function (event) {
            var target = event.target;
            if (target && target.matches && target.matches('select[id$="-arch"]')) {
                syncRow(rowFor(target));
            }
            if (target && target.matches && target.matches('select[id$="-xm_rfdetr_variant"]')) {
                fillRfdetrResolution(target);
            }
        });

        // Django 4.1+ dispatches a native CustomEvent on the freshly added row
        // (it bubbles to document) after "Add another".
        document.addEventListener("formset:added", function (event) {
            var target = event.target;
            if (target && target.closest) {
                syncRow(target.closest(".inline-related") || target);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
