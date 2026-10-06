class AppError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class NotFoundError(AppError):
    def __init__(self, what: str = "Image"):
        super().__init__(404, "not_found", f"{what} not found")
