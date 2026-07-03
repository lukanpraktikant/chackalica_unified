"""Admin form for :class:`~training.models.ExperimentModel`.

The DB row stays model-agnostic — ``arch`` is a string and every builder kwarg
lives in the ``params`` JSON. This form is the human-facing layer over that JSON:
it renders a real widget per builder option (see :mod:`training.model_specs`) for
*every* architecture, and JavaScript (``experiment_model_form.js``) shows only the
ones belonging to the currently selected ``arch``. On save the selected arch's
values are folded back into ``params``.

Fields for the non-selected archs are still submitted but ignored: :meth:`save`
only reads the specs for the chosen ``arch``, and first strips every spec-owned
key so switching arch never leaves a stale kwarg a different adapter would reject.
"""

from django import forms

from training import model_specs
from training.models import ExperimentModel


def _build_field(spec: dict, arch: str) -> forms.Field:
    """One form field for a spec, tagged so the JS can show/hide it by arch."""
    kind = spec["kind"]
    label = spec.get("label", spec["key"])
    help_text = spec.get("help", "")
    default = spec.get("default")
    if default is not None and kind != "bool":
        help_text = (help_text + f" (adapter default: {default})").strip()

    attrs = {"class": "xm-spec-field", "data-arch": arch}
    common = {"required": False, "label": label, "help_text": help_text}

    if kind == "choice":
        choices = [("", "(default)")] + model_specs.normalized_choices(spec)
        return forms.ChoiceField(
            choices=choices, widget=forms.Select(attrs=attrs), **common
        )
    if kind == "int":
        return forms.IntegerField(widget=forms.NumberInput(attrs=attrs), **common)
    if kind == "float":
        return forms.FloatField(
            widget=forms.NumberInput(attrs={**attrs, "step": "any"}), **common
        )
    if kind == "bool":
        # NullBooleanSelect gives a three-way Unknown/Yes/No; Unknown = adapter default.
        return forms.NullBooleanField(widget=forms.NullBooleanSelect(attrs=attrs), **common)
    return forms.CharField(widget=forms.TextInput(attrs=attrs), **common)


def _spec_fields() -> dict:
    """A declared form field per builder option of every arch.

    Declared at class-definition time (below) so the fields land in
    ``base_fields``/``declared_fields``: Django admin only renders — and the
    inline formset factory only accepts — fields declared on the form class,
    never ones added dynamically in ``__init__``.
    """
    return {
        model_specs.field_name(arch, spec["key"]): _build_field(spec, arch)
        for arch, specs in model_specs.ARCH_FIELD_SPECS.items()
        for spec in specs
    }


class ExperimentModelForm(forms.ModelForm):
    class Meta:
        model = ExperimentModel
        fields = ["arch", "pretrained", "num_classes", "params"]

    # Inject the per-option widgets into the class namespace so the metaclass
    # picks them up as declared fields (see _spec_fields).
    locals().update(_spec_fields())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = dict(getattr(self.instance, "params", None) or {})
        arch = self.instance.arch or ""

        # Seed each field of the *selected* arch with the stored value; only the
        # selected arch's fields are ever read back, so cross-arch key sharing
        # (e.g. "variant") is harmless.
        for spec in model_specs.ARCH_FIELD_SPECS.get(arch, []):
            if spec["key"] not in params:
                continue
            field = self.fields[model_specs.field_name(arch, spec["key"])]
            stored = params[spec["key"]]
            if spec["kind"] == "choice":
                stored_str = str(stored)
                if stored_str not in [c[0] for c in field.choices]:
                    # Preserve a hand-set value not in our list (e.g. a custom
                    # rtdetr weights repo) as a selectable option.
                    field.choices = list(field.choices) + [
                        (stored_str, f"(custom) {stored_str}")
                    ]
                field.initial = stored_str
            else:
                field.initial = stored

        # The raw JSON stays for open-ended ModelConfig kwargs the specs don't cover.
        self.fields["params"].help_text = (
            "Advanced: extra architecture kwargs as JSON. The fields above override "
            "any matching keys here on save."
        )

    def save(self, commit=True):
        obj = super().save(commit=False)
        arch = self.cleaned_data.get("arch") or ""

        params = dict(self.cleaned_data.get("params") or {})
        # Drop every spec-owned key, then re-apply only the selected arch's values,
        # so options from a previously selected arch don't linger.
        for key in model_specs.ALL_SPEC_KEYS:
            params.pop(key, None)
        for spec in model_specs.ARCH_FIELD_SPECS.get(arch, []):
            fname = model_specs.field_name(arch, spec["key"])
            value = self.cleaned_data.get(fname)
            if value in (None, ""):
                continue  # blank / "(default)" → let the adapter default apply
            params[spec["key"]] = value
        obj.params = params

        if commit:
            obj.save()
            self.save_m2m()
        return obj
