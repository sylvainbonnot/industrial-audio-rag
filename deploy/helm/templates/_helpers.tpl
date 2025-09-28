{{/*
Expand the name of the chart.
*/}}
{{- define "industrial-audio-rag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "industrial-audio-rag.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "industrial-audio-rag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "industrial-audio-rag.labels" -}}
helm.sh/chart: {{ include "industrial-audio-rag.chart" . }}
{{ include "industrial-audio-rag.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "industrial-audio-rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "industrial-audio-rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "industrial-audio-rag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "industrial-audio-rag.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Qdrant labels
*/}}
{{- define "industrial-audio-rag.qdrant.labels" -}}
helm.sh/chart: {{ include "industrial-audio-rag.chart" . }}
{{ include "industrial-audio-rag.qdrant.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Qdrant selector labels
*/}}
{{- define "industrial-audio-rag.qdrant.selectorLabels" -}}
app.kubernetes.io/name: {{ include "industrial-audio-rag.name" . }}-qdrant
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Ollama labels
*/}}
{{- define "industrial-audio-rag.ollama.labels" -}}
helm.sh/chart: {{ include "industrial-audio-rag.chart" . }}
{{ include "industrial-audio-rag.ollama.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: llm
{{- end }}

{{/*
Ollama selector labels
*/}}
{{- define "industrial-audio-rag.ollama.selectorLabels" -}}
app.kubernetes.io/name: {{ include "industrial-audio-rag.name" . }}-ollama
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}