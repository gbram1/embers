{{- define "embers.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "embers.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "embers.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "embers.labels" -}}
app.kubernetes.io/name: {{ include "embers.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "embers.selectorLabels" -}}
app.kubernetes.io/name: {{ include "embers.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
