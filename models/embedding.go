package models

import (
	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"
)

type TextEmbedding struct {
	ID        uint            `gorm:"primaryKey" json:"id"`
	Text      string          `gorm:"type:text" json:"text"`
	Embedding pgvector.Vector `gorm:"type:vector(768)" json:"embedding"`
}

func MigrateDB(db *gorm.DB) {
	db.AutoMigrate(&TextEmbedding{})

	db.Exec("CREATE INDEX IF NOT EXISTS ON text_embeddings USING hnsw (embedding vector_cosine_ops);")
}
