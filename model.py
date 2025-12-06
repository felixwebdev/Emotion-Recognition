import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

def build_cnn(input_shape=(48, 48, 1), num_classes=7):
    input_tensor = layers.Input(shape=input_shape)
    
    # Data Augmentation (Giữ nguyên)
    x = layers.RandomFlip("horizontal")(input_tensor)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x) # Thêm zoom nhẹ

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classification head
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    
    # Dùng learning rate nhỏ hơn một chút (0.0005 thay vì mặc định 0.001) để ổn định
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model