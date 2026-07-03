// Show only the builder-option fields (rendered by ExperimentModelForm) that
// belong to the architecture currently selected in each ExperimentModel inline
// row. Fields are tagged with class "xm-spec-field" and data-arch="<arch>"; we
// toggle the enclosing admin .form-row so labels/help hide with the widget.
(function ($) {
    "use strict";

    function syncRow(row) {
        var $row = $(row);
        var $arch = $row.find('select[id$="-arch"]').first();
        if (!$arch.length) {
            return;
        }
        var arch = $arch.val();
        $row.find(".xm-spec-field").each(function () {
            var matches = this.getAttribute("data-arch") === arch;
            var $formRow = $(this).closest(".form-row");
            ($formRow.length ? $formRow : $(this)).toggle(matches);
        });
    }

    function rowFor(el) {
        var $rel = $(el).closest(".inline-related");
        return $rel.length ? $rel : $(el).closest("tr, .form-row").parent();
    }

    $(function () {
        // Skip the hidden empty-form template; real rows get cloned from it.
        $(".inline-related").not(".empty-form").each(function () {
            syncRow(this);
        });

        $(document).on("change", 'select[id$="-arch"]', function () {
            syncRow(rowFor(this));
        });

        // Django fires this (jQuery event) after "Add another" clones a row.
        $(document).on("formset:added", function (event, $row) {
            syncRow(($row && $row.length ? $row : $(event.target)));
        });
    });
})(typeof django !== "undefined" ? django.jQuery : window.jQuery);
