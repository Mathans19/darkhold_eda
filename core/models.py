from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Advanced Metadata
    chaos_score = models.IntegerField(default=50) # 0-100
    total_rows = models.IntegerField(default=0)
    total_cols = models.IntegerField(default=0)
    ai_summary = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name or str(self.file)
