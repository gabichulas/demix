resource "aws_s3_bucket" "raw_audio" {
  bucket = "demix-raw-audio"
}

resource "aws_sqs_queue" "audio_tasks" {
  name = "demix-processing-queue"
}